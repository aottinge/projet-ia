"""Étape 3 : clustering K-Means des fiches produites par le LLM, expériences, sauvegarde du modèle.

    python 3_entrainer_clustering.py          # k choisi par la silhouette entre 4 et 10
    python 3_entrainer_clustering.py --k 6    # k imposé (après avoir regardé coude et silhouette)

Sorties :
  modeles/clustering.joblib      pipeline (vectoriseur + K-Means), ACP, seuil « recette atypique »
  modeles/recettes_train.pkl     recettes d'entraînement avec leur famille (pour l'application)
  modeles/familles.json          portrait de chaque famille
  resultats/                     figures et tableaux des expériences
"""
import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline

from config import ECHANTILLON, FAMILLES, FICHES, GRAINE, MODELES, PIPELINE, RECETTES_TRAIN, RESULTATS
from features import COUTS, VectoriseurFiche
from llm import lire_fiches
from recettes import lire_echantillon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=None)
args = parser.parse_args()
MODELES.mkdir(exist_ok=True)
RESULTATS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. Données : recettes + fiches LLM
df = lire_echantillon(ECHANTILLON)
fiches = lire_fiches(FICHES)
print(f"{len(fiches)} fiches LLM pour {len(df)} recettes")
if len(fiches) < 50:
    raise SystemExit("Pas assez de fiches : lancez d'abord 2_extraire_fiches.py")
df = df[df["RecipeId"].isin(fiches)].reset_index(drop=True)
df = pd.concat([df, pd.DataFrame([fiches[i] for i in df["RecipeId"]])], axis=1)
train, test = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
print(f"entraînement : {len(train)} recettes | test : {len(test)} recettes")

vect = VectoriseurFiche().fit(train)
X_train, X_test = vect.transform(train), vect.transform(test)
print(f"{X_train.shape[1]} variables, dont {len(vect.vocabulaire_)} ingrédients :", ", ".join(vect.vocabulaire_))

# ---------------------------------------------------------------- 2. Choisir k : coude et silhouette
ks = list(range(2, 13))
inerties, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit(X_train)
    inerties.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train, km.labels_))
choix = pd.DataFrame({"k": ks, "inertie": inerties, "silhouette": silhouettes}).round(3)
print("\n" + choix.to_string(index=False))
k = args.k or int(choix[choix["k"].between(4, 10)].sort_values("silhouette").iloc[-1]["k"])
print(f"\nk retenu : {k}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ks, inerties, "o-")
axes[0].set(title="Méthode du coude", xlabel="k", ylabel="inertie")
axes[1].plot(ks, silhouettes, "o-", color="tab:orange")
axes[1].set(title="Score de silhouette (entraînement)", xlabel="k", ylabel="silhouette")
for ax in axes:
    ax.axvline(k, color="grey", ls="--")
fig.tight_layout()
fig.savefig(RESULTATS / "choix_k.png", dpi=120)

# ---------------------------------------------------------------- 3. Le modèle final
pipeline = Pipeline([("vectoriseur", VectoriseurFiche()),
                     ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=GRAINE))]).fit(train)
train["famille"] = pipeline.predict(train)
test["famille"] = pipeline.predict(test)
X_train = pipeline[0].transform(train)
distances = pipeline[-1].transform(X_train)[np.arange(len(train)), train["famille"]]
seuil_atypique = float(np.quantile(distances, 0.95))
train["distance_centre"] = distances

# ---------------------------------------------------------------- 4. Expériences (sur le test)
def evaluer(nom, labels_test, X_test_espace=None, labels_train=None):
    ligne = {"représentation": nom,
             "AMI test (vs catégorie Food.com)": adjusted_mutual_info_score(test["categorie"], labels_test),
             "ARI test (vs catégorie Food.com)": adjusted_rand_score(test["categorie"], labels_test)}
    if X_test_espace is not None:
        ligne["silhouette test (dans son espace)"] = silhouette_score(X_test_espace, labels_test)
    if labels_train is not None:
        ligne["AMI entraînement"] = adjusted_mutual_info_score(train["categorie"], labels_train)
    return ligne


lignes = [evaluer("A. fiches LLM, colonnes 0/1 non standardisées (modèle retenu)",
                  test["famille"], pipeline[0].transform(test), train["famille"])]

pipe_std = Pipeline([("vectoriseur", VectoriseurFiche(standardiser_tout=True)),
                     ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=GRAINE))]).fit(train)
lignes.append(evaluer("B. fiches LLM, tout standardisé", pipe_std.predict(test),
                      pipe_std[0].transform(test), pipe_std.predict(train)))

# Référence sans LLM : TF-IDF sur la liste complète des ingrédients fournie par Food.com
tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, min_df=5)
T_train = tfidf.fit_transform(train["ingredients_reels"])
T_test = tfidf.transform(test["ingredients_reels"])
km_tfidf = KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit(T_train)
lignes.append(evaluer("C. sans LLM : TF-IDF de la liste d'ingrédients Food.com", km_tfidf.predict(T_test),
                      T_test, km_tfidf.labels_))

rng = np.random.default_rng(GRAINE)
lignes.append(evaluer("D. familles tirées au hasard (plancher)", rng.integers(0, k, len(test))))
comparaison = pd.DataFrame(lignes).round(3)
print("\n" + comparaison.to_string(index=False))

# Stabilité : relancer K-Means avec 5 graines et comparer les découpages entre eux (ARI)
decoupages = [KMeans(n_clusters=k, n_init=1, random_state=g).fit_predict(X_train) for g in range(5)]
stabilite = float(np.mean([adjusted_rand_score(a, b) for i, a in enumerate(decoupages) for b in decoupages[i + 1:]]))
print(f"\nStabilité (ARI moyen entre 5 initialisations uniques) : {stabilite:.3f}")

# ---------------------------------------------------------------- 5. Portrait de chaque famille
frequence_globale = pd.Series([i for l in train["ingredients_principaux"] for i in l]).value_counts() / len(train)
familles = []
for f, groupe in train.groupby("famille"):
    freq = pd.Series([i for l in groupe["ingredients_principaux"] for i in l]).value_counts() / len(groupe)
    lift = (freq / frequence_globale[freq.index]).where(freq >= 0.2).dropna().sort_values(ascending=False)
    typiques = lift.head(3).index.tolist() or freq.head(3).index.tolist()
    mode_dominant = groupe["mode_cuisson"].value_counts().idxmax()
    familles.append({
        "famille": int(f),
        "nom_auto": f"{', '.join(typiques)} — {mode_dominant}",
        "effectif": int(len(groupe)),
        "temps_median_minutes": float(groupe["temps_total_minutes"].median()),
        "couts": groupe["cout_estime"].value_counts(normalize=True).reindex(COUTS, fill_value=0).round(2).to_dict(),
        "modes_cuisson": groupe["mode_cuisson"].value_counts(normalize=True).round(2).head(3).to_dict(),
        "ingredients_frequents": freq.head(8).round(2).to_dict(),
        "ingredients_typiques": typiques,
        "categories_food_com": groupe["categorie"].value_counts(normalize=True).round(2).head(3).to_dict(),
        "exemples": groupe.nsmallest(6, "distance_centre")["nom"].tolist(),
    })
    print(f"\nFamille {f} ({len(groupe)} recettes) : {familles[-1]['nom_auto']}")
    print("   ", " / ".join(familles[-1]["exemples"][:4]))
FAMILLES.write_text(json.dumps(familles, ensure_ascii=False, indent=2), encoding="utf-8")

pd.crosstab(test["famille"], test["categorie"]).to_csv(RESULTATS / "familles_vs_categories_test.csv")

# ---------------------------------------------------------------- 6. ACP pour la visualisation
pca = PCA(n_components=2, random_state=GRAINE).fit(X_train)
coords = pca.transform(X_train)
train["pca_x"], train["pca_y"] = coords[:, 0], coords[:, 1]
print(f"\nACP : les deux axes conservent {pca.explained_variance_ratio_.sum():.0%} de la variance")
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(coords[:, 0], coords[:, 1], c=train["famille"], cmap="tab10", s=12, alpha=0.7)
ax.legend(*sc.legend_elements(), title="famille", loc="best", fontsize=8)
ax.set(title=f"Familles de recettes (ACP, {pca.explained_variance_ratio_.sum():.0%} de la variance)",
       xlabel="axe 1", ylabel="axe 2")
fig.tight_layout()
fig.savefig(RESULTATS / "acp_familles.png", dpi=120)

# ---------------------------------------------------------------- 7. Sauvegardes
joblib.dump({"pipeline": pipeline, "pca": pca, "seuil_atypique": seuil_atypique, "k": k,
             "X_train": X_train}, PIPELINE)
train.to_pickle(RECETTES_TRAIN)
choix.to_csv(RESULTATS / "choix_k.csv", index=False)
comparaison.to_csv(RESULTATS / "comparaison_representations.csv", index=False)
(RESULTATS / "clustering.json").write_text(json.dumps({
    "k": k, "n_train": len(train), "n_test": len(test), "n_variables": int(X_train.shape[1]),
    "vocabulaire": pipeline[0].vocabulaire_, "stabilite_ari": stabilite,
    "seuil_atypique": seuil_atypique, "variance_acp": float(pca.explained_variance_ratio_.sum()),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nModèle et résultats enregistrés dans modeles/ et resultats/")

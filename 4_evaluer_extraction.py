"""Étape 4 : évaluer les fiches produites par le LLM, et comparer zéro exemple / few-shot.

    python 4_evaluer_extraction.py
    python 4_evaluer_extraction.py data/fiches_llm.jsonl data/autre_modele.jsonl

On compare chaque champ de la fiche à une vérité terrain, sur les recettes de TEST :
  - temps_total_minutes   <-> TotalTime de Food.com (exact, saisi par l'auteur de la recette)
  - mode_cuisson          <-> mots-clés Food.com « Oven », « Stove Top », « No Cook »… (quand il n'y en a qu'un)
  - ingredients_principaux <-> liste complète des ingrédients : un ingrédient extrait qui n'y figure pas est inventé
  - cout_estime           <-> pas de vérité dans Food.com : annotations faites à la main dans
                              data/annotations_cout.csv, et indice faible du mot-clé « Inexpensive »
"""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from config import DATA, ECHANTILLON, FICHES, FICHES_FEW_SHOT, GRAINE, RACINE, RESULTATS
from features import COUTS, MODES
from llm import lire_fiches
from recettes import lire_echantillon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANNOTATIONS = DATA / "annotations_cout.csv"
RESULTATS.mkdir(exist_ok=True)

fichiers = [Path(a) if Path(a).is_absolute() else RACINE / a for a in sys.argv[1:]] \
    or [f for f in (FICHES, FICHES_FEW_SHOT) if f.exists()]
df = lire_echantillon(ECHANTILLON)
test = df[df["split"] == "test"].set_index("RecipeId")

# Modèle de fichier d'annotations : 40 recettes de test, colonne à remplir à la main
if not ANNOTATIONS.exists():
    modele = test.sample(40, random_state=GRAINE).reset_index()[["RecipeId", "nom", "texte"]]
    modele["cout_reference"] = ""
    modele.to_csv(ANNOTATIONS, index=False)
    print(f"Fichier {ANNOTATIONS.name} créé : remplissez la colonne cout_reference (faible / moyen / élevé)")
annotations = pd.read_csv(ANNOTATIONS).dropna(subset=["cout_reference"]).set_index("RecipeId")["cout_reference"]
annotations = annotations[annotations.isin(COUTS)]


def mots(texte):
    """Mots d'un nom d'ingrédient, au singulier (approximatif : on retire le s final)."""
    return {re.sub(r"(?<=[a-z]{3})(es|s)$", "", m) for m in re.findall(r"[a-z]+", texte.lower())}


def ingredient_trouve(extrait, reels):
    """Vrai si tous les mots de l'ingrédient extrait figurent dans un des ingrédients réels."""
    cible = mots(extrait)
    return bool(cible) and any(cible <= mots(r) for r in reels)


def evaluer(fiches, ids, nom):
    lignes = []
    for rid in ids:
        f, vrai = fiches[rid], test.loc[rid]
        trouves = [ingredient_trouve(i, vrai["ingredients_reels"]) for i in f["ingredients_principaux"]]
        lignes.append({"RecipeId": rid, "nom": vrai["nom"],
                       "temps_reel": vrai["temps_reel_minutes"], "temps_llm": f["temps_total_minutes"],
                       "mode_reel": vrai["mode_cuisson_mots_cles"], "mode_llm": f["mode_cuisson"],
                       "inexpensive": vrai["mot_cle_inexpensive"], "cout_llm": f["cout_estime"],
                       "n_ingr": len(trouves), "n_ingr_trouves": sum(trouves),
                       "ingr_inventes": [i for i, t in zip(f["ingredients_principaux"], trouves) if not t]})
    r = pd.DataFrame(lignes).set_index("RecipeId")
    ecart = (r["temps_llm"] - r["temps_reel"]).abs()
    rapport = r["temps_llm"] / r["temps_reel"]
    avec_mode = r.dropna(subset=["mode_reel"])
    avec_annot = r.join(annotations, how="inner")
    resume = {
        "fiches": nom, "recettes évaluées": len(r),
        "temps : erreur absolue médiane (min)": ecart.median(),
        "temps : erreur absolue moyenne (min)": ecart.mean(),
        "temps : part à ±25 %": rapport.between(0.75, 1.25).mean(),
        "temps : part sous-estimée de plus de moitié": (rapport < 0.5).mean(),
        "mode de cuisson : exactitude": accuracy_score(avec_mode["mode_reel"], avec_mode["mode_llm"]),
        "mode de cuisson : recettes avec vérité": len(avec_mode),
        "ingrédients : précision (non inventés)": r["n_ingr_trouves"].sum() / r["n_ingr"].sum(),
        "ingrédients : nombre moyen extrait": r["n_ingr"].mean(),
        "coût « faible » si mot-clé Inexpensive": (r.loc[r["inexpensive"], "cout_llm"] == "faible").mean(),
        "coût « faible » sinon": (r.loc[~r["inexpensive"], "cout_llm"] == "faible").mean(),
        "coût : exactitude vs annotations": (accuracy_score(avec_annot["cout_reference"], avec_annot["cout_llm"])
                                             if len(avec_annot) else np.nan),
        "coût : recettes annotées": len(avec_annot),
    }
    return resume, r, avec_mode


toutes = {f.stem: lire_fiches(f) for f in fichiers}
for nom, fiches in toutes.items():
    print(f"{nom} : {sum(i in test.index for i in fiches)} fiches de test")
# Comparaison équitable : seulement les recettes de test présentes dans tous les fichiers
communs = sorted(set(test.index).intersection(*[set(f) for f in toutes.values()]))
print(f"Recettes de test communes à tous les fichiers : {len(communs)}\n")
if not communs:
    raise SystemExit("Aucune fiche de test : lancez d'abord 2_extraire_fiches.py")

resumes = []
for nom, fiches in toutes.items():
    resume, details, avec_mode = evaluer(fiches, communs, nom)
    resumes.append(resume)
    details.to_csv(RESULTATS / f"extraction_details_{nom}.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(avec_mode["mode_reel"], avec_mode["mode_llm"], labels=MODES,
                                            cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False)
    ax.set(title=f"Mode de cuisson — {nom}", xlabel="prédit par le LLM", ylabel="mots-clés Food.com")
    fig.tight_layout()
    fig.savefig(RESULTATS / f"confusion_mode_{nom}.png", dpi=120)
    print(f"--- {nom} : les 5 plus grosses erreurs de temps")
    pire = details.assign(ecart=(details["temps_llm"] - details["temps_reel"]).abs()).nlargest(5, "ecart")
    print(pire[["nom", "temps_reel", "temps_llm"]].to_string(), "\n")
    inventes = pd.Series([i for l in details["ingr_inventes"] for i in l]).value_counts().head(10)
    print(f"--- {nom} : ingrédients extraits absents de la recette (les plus fréquents)\n"
          + (inventes.to_string() if len(inventes) else "aucun") + "\n")

tableau = pd.DataFrame(resumes).set_index("fiches").T
pd.set_option("display.float_format", "{:.3f}".format)
print(tableau.to_string())
tableau.to_csv(RESULTATS / "extraction_comparaison.csv")
print(f"\nRésultats enregistrés dans {RESULTATS.name}/")

"""Application « Familles de recettes » : streamlit run app.py

1. l'utilisateur colle une recette en texte libre ;
2. le LLM la transforme en fiche structurée (IA générative, sortie structurée) ;
3. le K-Means entraîné sur 1 000 fiches lui attribue une famille (IA prédictive, clustering).
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import ECHANTILLON, FAMILLES, NOMS_FAMILLES, PIPELINE, RECETTES_TRAIN, RESULTATS
from features import COUTS, MODES
from llm import FicheRecette, extraire_fiche, modele_par_defaut
from recettes import lire_echantillon

st.set_page_config(page_title="Familles de recettes", page_icon="🍲", layout="wide")


@st.cache_resource
def charger():
    """Chargé une seule fois au démarrage : l'application ne réentraîne jamais le modèle."""
    modele = joblib.load(PIPELINE)
    train = pd.read_pickle(RECETTES_TRAIN)
    familles = {f["famille"]: f for f in json.loads(FAMILLES.read_text(encoding="utf-8"))}
    noms = json.loads(NOMS_FAMILLES.read_text(encoding="utf-8")) if NOMS_FAMILLES.exists() else {}
    for f, d in familles.items():
        d["nom"] = noms.get(str(f), d["nom_auto"])
    test = lire_echantillon(ECHANTILLON).query("split == 'test'")
    return modele, train, familles, test


if not PIPELINE.exists():
    st.error("Modèle introuvable : lancez d'abord les scripts 1 à 3 (voir le README).")
    st.stop()
modele, train, familles, test = charger()
pipeline, pca, seuil = modele["pipeline"], modele["pca"], modele["seuil_atypique"]
vect, kmeans = pipeline[0], pipeline[-1]
COULEURS = plt.get_cmap("tab10")


def nom_famille(f):
    return f"Famille {f} · {familles[f]['nom']}"


def graphique_acp(point=None, famille=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for f in sorted(familles):
        sel = train["famille"] == f
        ax.scatter(train.loc[sel, "pca_x"], train.loc[sel, "pca_y"], s=8, alpha=0.35 if point is not None else 0.6,
                   color=COULEURS(f % 10), label=f"famille {f}")
    if point is not None:
        ax.scatter(*point, marker="*", s=400, color=COULEURS(famille % 10), edgecolor="black", zorder=5,
                   label="votre recette")
    ax.legend(fontsize=7, loc="best")
    ax.set(xlabel="axe 1 de l'ACP", ylabel="axe 2 de l'ACP",
           title=f"Projection ACP ({pca.explained_variance_ratio_.sum():.0%} de la variance conservée)")
    fig.tight_layout()
    return fig


def portrait(f):
    d = familles[f]
    c1, c2, c3 = st.columns(3)
    c1.metric("Recettes", d["effectif"])
    c2.metric("Temps médian", f"{d['temps_median_minutes']:.0f} min")
    c3.metric("Coût le plus fréquent", max(d["couts"], key=d["couts"].get))
    st.markdown("**Ingrédients typiques** (sur-représentés dans la famille) : " + ", ".join(d["ingredients_typiques"]))
    st.markdown("**Ingrédients fréquents** : " + ", ".join(f"{i} ({p:.0%})" for i, p in d["ingredients_frequents"].items()))
    st.markdown("**Modes de cuisson** : " + ", ".join(f"{m} ({p:.0%})" for m, p in d["modes_cuisson"].items()))
    st.markdown("**Catégories Food.com** (non utilisées par le clustering) : "
                + ", ".join(f"{c} ({p:.0%})" for c, p in d["categories_food_com"].items()))
    st.markdown("**Recettes les plus représentatives** (proches du centre) : " + " · ".join(d["exemples"]))


st.title("🍲 Familles de recettes")
st.caption(f"LLM : `{modele_par_defaut()}` (sortie structurée) · K-Means à {modele['k']} familles "
           f"entraîné sur {len(train)} recettes Food.com")
onglet_classer, onglet_familles, onglet_exp = st.tabs(["Classer une recette", "Les familles", "Expériences"])

# ------------------------------------------------------------------------------------------ Classer
with onglet_classer:
    if "texte" not in st.session_state:
        st.session_state.texte = test["texte"].iloc[0]
    c_gauche, c_droite = st.columns([1, 1])
    with c_gauche:
        if st.button("🎲 Prendre une recette de test au hasard"):
            st.session_state.texte = test["texte"].sample(1).iloc[0]
            st.session_state.pop("fiche", None)
        texte = st.text_area("Recette en texte libre (anglais ou français)", key="texte", height=330)
        few_shot = st.checkbox("Prompt avec deux exemples (few-shot)", value=False)
        if st.button("Analyser la recette", type="primary"):
            try:
                with st.spinner("Le LLM remplit la fiche…"):
                    fiche, reponse = extraire_fiche(texte, few_shot=few_shot)
                st.session_state.fiche = fiche.model_dump()
                st.session_state.tokens = (reponse.usage.prompt_tokens, reponse.usage.completion_tokens)
            except Exception as e:
                st.error(f"Échec de l'extraction : {e}")

    with c_droite:
        if "fiche" in st.session_state:
            st.subheader("1. La fiche produite par le LLM")
            st.json(st.session_state.fiche)
            st.caption("Tokens : {} envoyés, {} reçus".format(*st.session_state.tokens))

            with st.expander("✏️ Modifier la fiche à la main (et si… ?)"):
                fiche_llm = st.session_state.fiche
                ingr = st.multiselect("Ingrédients principaux (hors vocabulaire : ignorés par le modèle)",
                                      sorted(set(vect.vocabulaire_) | set(fiche_llm["ingredients_principaux"])),
                                      default=fiche_llm["ingredients_principaux"])
                temps = st.slider("Temps total (min)", 1, 1440, min(fiche_llm["temps_total_minutes"], 1440))
                cout = st.select_slider("Coût", COUTS, value=fiche_llm["cout_estime"])
                mode = st.selectbox("Mode de cuisson", MODES, index=MODES.index(fiche_llm["mode_cuisson"]))
                fiche = FicheRecette(ingredients_principaux=ingr, temps_total_minutes=temps,
                                     cout_estime=cout, mode_cuisson=mode).model_dump()
            hors_vocab = [i for i in fiche["ingredients_principaux"] if i not in vect.vocabulaire_]

    if "fiche" in st.session_state:
        st.divider()
        x = vect.transform([fiche])
        dist = kmeans.transform(x)[0]
        f = int(np.argmin(dist))
        st.subheader(f"2. Famille prédite : {nom_famille(f)}")
        if dist[f] > seuil:
            st.warning(f"Recette atypique : sa distance au centre de la famille ({dist[f]:.2f}) dépasse celle de "
                       f"95 % des recettes d'entraînement ({seuil:.2f}). K-Means l'affecte quand même à la "
                       "famille la plus proche, mais elle ne ressemble vraiment à aucune.")
        if hors_vocab:
            st.info("Ingrédients inconnus du modèle (absents du vocabulaire appris), donc ignorés : "
                    + ", ".join(hors_vocab))
        g1, g2 = st.columns([1, 1])
        with g1:
            ordre = np.argsort(dist)
            st.markdown("**Distance au centre de chaque famille** (plus petit = plus proche)")
            st.bar_chart(pd.DataFrame({"distance": dist[ordre]},
                                      index=[nom_famille(i)[:45] for i in ordre]), horizontal=True)
            ecart = dist[ordre[1]] - dist[ordre[0]]
            st.caption(f"Écart avec la 2ᵉ famille la plus proche : {ecart:.2f} "
                       + ("(hésitation : recette à la frontière)" if ecart < 0.3 else ""))
            voisins = np.argsort(np.linalg.norm(modele["X_train"] - x, axis=1))[:5]
            st.markdown("**Les 5 recettes d'entraînement les plus proches**")
            st.dataframe(train.iloc[voisins][["nom", "famille", "temps_total_minutes", "mode_cuisson",
                                               "ingredients_principaux"]], hide_index=True)
        with g2:
            st.pyplot(graphique_acp(pca.transform(x)[0], f))
        with st.expander(f"Portrait de la famille {f}", expanded=True):
            portrait(f)

# ------------------------------------------------------------------------------------------ Familles
with onglet_familles:
    st.pyplot(graphique_acp(), width="content")
    st.caption("L'ACP ne sert qu'à dessiner : le K-Means est calculé sur toutes les variables. Deux familles "
               "qui se chevauchent sur le dessin peuvent être bien séparées dans l'espace complet.")
    for f in sorted(familles):
        with st.expander(nom_famille(f)):
            portrait(f)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=vect.get_feature_names_out()).round(2)
    with st.expander("Centroïdes (temps et coût standardisés, autres colonnes = proportion de recettes)"):
        st.dataframe(centres.T)

# ------------------------------------------------------------------------------------------ Expériences
with onglet_exp:
    st.subheader("Clustering : choix de k")
    if (RESULTATS / "choix_k.png").exists():
        st.image(str(RESULTATS / "choix_k.png"))
        st.dataframe(pd.read_csv(RESULTATS / "choix_k.csv"), hide_index=True)
    st.subheader("Clustering : quelle représentation des recettes ?")
    if (RESULTATS / "comparaison_representations.csv").exists():
        st.dataframe(pd.read_csv(RESULTATS / "comparaison_representations.csv"), hide_index=True)
        st.caption("AMI / ARI : accord entre les familles et les catégories Food.com sur les recettes de test "
                   "(0 = aucun lien, 1 = identiques). Les silhouettes ne se comparent pas d'une représentation "
                   "à l'autre : elles sont calculées dans des espaces différents.")
    st.subheader("Sortie structurée : qualité de l'extraction (recettes de test)")
    if (RESULTATS / "extraction_comparaison.csv").exists():
        st.dataframe(pd.read_csv(RESULTATS / "extraction_comparaison.csv", index_col=0))
        for png in sorted(RESULTATS.glob("confusion_mode_*.png")):
            st.image(str(png), width=480)
    else:
        st.info("Lancez 4_evaluer_extraction.py pour afficher cette partie.")

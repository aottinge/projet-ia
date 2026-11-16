"""Étape 1 : tirer un échantillon de recettes Food.com et construire leur texte libre.

    python 1_preparer_donnees.py            # 1 200 recettes (1 000 d'entraînement, 200 de test)
    python 1_preparer_donnees.py --n 600

Entrée : recipes.csv (Kaggle « Food.com - Recipes and Reviews », CC0), à placer à la racine.
Sortie : data/recettes_echantillon.csv
"""
import argparse

import numpy as np
import pandas as pd

from config import CSV_BRUT, DATA, ECHANTILLON, GRAINE
from recettes import duree_minutes, mode_cuisson_mots_cles, texte_recette, vecteur_r

COLONNES = ["RecipeId", "Name", "TotalTime", "RecipeCategory", "Keywords",
            "RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeInstructions", "ReviewCount"]

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1200, help="taille de l'échantillon")
parser.add_argument("--part-test", type=float, default=1 / 6, help="part des recettes gardées pour le test")
args = parser.parse_args()

print("Lecture de", CSV_BRUT.name, "(peut prendre une minute)…")
df = pd.read_csv(CSV_BRUT, usecols=COLONNES)
print(f"{len(df):,} recettes")

# 1. Filtrer : recettes notées par au moins 3 personnes (moins de recettes fantaisistes),
#    catégories qui décrivent un plat (on écarte les catégories de durée comme « < 30 Mins »).
df = df[df["ReviewCount"] >= 3]
df = df[~df["RecipeCategory"].fillna("< ").str.startswith("<")]
df["temps_reel_minutes"] = df["TotalTime"].map(duree_minutes)
df = df[df["temps_reel_minutes"].between(1, 48 * 60)]      # durées absentes ou aberrantes
# Food.com perd souvent des ingrédients (la levure d'un pain, par exemple) : la liste des noms est
# alors plus courte que celle des quantités. On ne garde que les recettes où les deux sont alignées
# (environ une sur quatre), pour que le texte soit complet et la vérité terrain fiable.
alignees = (df["RecipeIngredientParts"].map(lambda v: len(vecteur_r(v)))
            == df["RecipeIngredientQuantities"].map(lambda v: len(vecteur_r(v))))
df = df[alignees & df["RecipeIngredientParts"].notna()]
categories = df["RecipeCategory"].value_counts().head(20).index
df = df[df["RecipeCategory"].isin(categories)]
print(f"{len(df):,} recettes après filtrage, réparties dans {len(categories)} catégories")

# 2. Échantillon aléatoire (les proportions des catégories sont conservées)
ech = df.sample(n=args.n, random_state=GRAINE).reset_index(drop=True)

# 3. Texte libre et vérités terrain
ingredients = ech["RecipeIngredientParts"].map(vecteur_r)
quantites = ech["RecipeIngredientQuantities"].map(vecteur_r)
instructions = ech["RecipeInstructions"].map(vecteur_r)
mots_cles = ech["Keywords"].map(vecteur_r)

sortie = pd.DataFrame({
    "RecipeId": ech["RecipeId"],
    "nom": ech["Name"],
    "categorie": ech["RecipeCategory"],
    "temps_reel_minutes": ech["temps_reel_minutes"].astype(int),
    "mode_cuisson_mots_cles": mots_cles.map(mode_cuisson_mots_cles),
    "mot_cle_inexpensive": mots_cles.map(lambda m: "Inexpensive" in m),
    "ingredients_reels": [" | ".join(i for i in ing if i) for ing in ingredients],
    "texte": [texte_recette(n, i, q, e) for n, i, q, e in
              zip(ech["Name"], ingredients, quantites, instructions)],
})
rng = np.random.default_rng(GRAINE)
sortie["split"] = np.where(rng.random(len(sortie)) < args.part_test, "test", "train")

DATA.mkdir(exist_ok=True)
sortie.to_csv(ECHANTILLON, index=False)
print(f"\n{len(sortie)} recettes écrites dans {ECHANTILLON.relative_to(ECHANTILLON.parents[1])}")
print(sortie["split"].value_counts().to_string())
print("\nMode de cuisson connu par les mots-clés :", sortie["mode_cuisson_mots_cles"].notna().sum(), "recettes")
print("Longueur moyenne du texte :", int(sortie["texte"].str.len().mean()), "caractères")
print("\nExemple de texte envoyé au LLM :\n" + "-" * 60 + "\n" + sortie["texte"][0][:1200])

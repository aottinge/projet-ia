"""Lecture du format Food.com : vecteurs R, durées ISO 8601, texte libre d'une recette."""
import re

import pandas as pd

_DUREE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?")


def vecteur_r(valeur):
    """'c("a", "b", NA)' -> ['a', 'b', None] ; '"a"' -> ['a'] ; NaN -> []."""
    if not isinstance(valeur, str):
        return []
    valeur = valeur.strip()
    if not valeur.startswith("c("):
        return [valeur.strip('"')]
    elements = []
    for morceau in re.finditer(r'"((?:[^"\\]|\\.)*)"|NA', valeur[2:-1]):
        elements.append(morceau.group(1) if morceau.group(1) is not None else None)
    return elements


def duree_minutes(iso):
    """'PT1H30M' -> 90. Renvoie None si la durée est absente ou illisible."""
    if not isinstance(iso, str):
        return None
    m = _DUREE.fullmatch(iso)
    if not m or (m.group(1) is None and m.group(2) is None):
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


# Mots-clés Food.com qui désignent un mode de cuisson, et la valeur correspondante de notre schéma.
# Ils servent de vérité terrain (imparfaite) pour évaluer l'extraction du mode de cuisson.
MOTS_CLES_CUISSON = {
    "Oven": "four",
    "Stove Top": "plaque de cuisson",
    "Stir Fry": "plaque de cuisson",
    "Broil/Grill": "grill ou barbecue",
    "Deep Fried": "friture",
    "Pressure Cooker": "mijoteuse ou autocuiseur",
    "Microwave": "micro-ondes",
    "No Cook": "sans cuisson",
}


def mode_cuisson_mots_cles(mots_cles):
    """Mode de cuisson d'après les mots-clés, seulement s'il est sans ambiguïté (un seul mode cité)."""
    modes = {MOTS_CLES_CUISSON[m] for m in mots_cles if m in MOTS_CLES_CUISSON}
    return modes.pop() if len(modes) == 1 else None


def texte_recette(nom, ingredients, quantites, instructions):
    """Construit la recette « en texte libre » envoyée au LLM.

    On n'y met volontairement ni la durée, ni la catégorie, ni les mots-clés :
    le LLM doit les déduire du texte, comme pour une recette collée par un utilisateur.
    """
    if len(quantites) == len(ingredients):
        lignes = [f"- {q} {i}" if q else f"- {i}" for q, i in zip(quantites, ingredients)]
    else:  # les deux listes de Food.com ne sont pas toujours alignées : on ne garde que les noms
        lignes = [f"- {i}" for i in ingredients]
    etapes = [f"{n}. {e.strip()}" for n, e in enumerate(instructions, 1) if e]
    return f"{nom}\n\nIngredients:\n" + "\n".join(lignes) + "\n\nInstructions:\n" + "\n".join(etapes)


def lire_echantillon(chemin):
    """Relit l'échantillon en retransformant les listes stockées sous forme de texte."""
    df = pd.read_csv(chemin)
    df["ingredients_reels"] = df["ingredients_reels"].fillna("").str.split(" | ", regex=False)
    return df

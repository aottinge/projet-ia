"""Étape 2 : faire passer chaque recette de l'échantillon par le LLM (sortie structurée).

    python 2_extraire_fiches.py                       # toutes les recettes, zéro exemple
    python 2_extraire_fiches.py --limite 5            # pour tester sur 5 recettes d'abord
    python 2_extraire_fiches.py --few-shot --split test --sortie data/fiches_llm_few_shot.jsonl

Le fichier de sortie sert de cache : une recette déjà traitée n'est jamais renvoyée au LLM,
on peut donc interrompre le script (Ctrl+C) et le relancer.
"""
import argparse
import json
import time
from pathlib import Path

from config import ECHANTILLON, FICHES, RACINE
from llm import extraire_fiche, lire_fiches, modele_par_defaut
from recettes import lire_echantillon

parser = argparse.ArgumentParser()
parser.add_argument("--few-shot", action="store_true", help="ajouter deux exemples résolus au prompt")
parser.add_argument("--modele", default=None, help="nom du modèle (par défaut : LLM_MODEL du .env)")
parser.add_argument("--split", choices=["train", "test", "tout"], default="tout")
parser.add_argument("--limite", type=int, default=None, help="nombre maximal de recettes à traiter")
parser.add_argument("--sortie", type=Path, default=FICHES)
args = parser.parse_args()
sortie = args.sortie if args.sortie.is_absolute() else RACINE / args.sortie

df = lire_echantillon(ECHANTILLON)
if args.split != "tout":
    df = df[df["split"] == args.split]
deja = set(lire_fiches(sortie))
a_faire = df[~df["RecipeId"].isin(deja)]
if args.limite:
    a_faire = a_faire.head(args.limite)
modele = args.modele or modele_par_defaut()
print(f"Modèle : {modele} | few-shot : {args.few_shot} | déjà faites : {len(deja)} | à faire : {len(a_faire)}")

tokens_entree = tokens_sortie = erreurs = 0
debut = time.perf_counter()
with sortie.open("a", encoding="utf-8") as f:
    for n, (recipe_id, nom, texte) in enumerate(a_faire[["RecipeId", "nom", "texte"]].itertuples(index=False), 1):
        enregistrement = {"RecipeId": int(recipe_id), "modele": modele, "few_shot": args.few_shot}
        try:
            fiche, reponse = extraire_fiche(texte, few_shot=args.few_shot, modele=modele)
            enregistrement["fiche"] = fiche.model_dump()
            tokens_entree += reponse.usage.prompt_tokens
            tokens_sortie += reponse.usage.completion_tokens
        except Exception as e:  # on note l'erreur et on continue : une recette ratée ne bloque pas les autres
            enregistrement["erreur"] = f"{type(e).__name__}: {e}"
            erreurs += 1
        f.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
        f.flush()
        if n % 25 == 0 or n == len(a_faire):
            duree = time.perf_counter() - debut
            print(f"{n}/{len(a_faire)} | {duree / n:.2f} s par recette | erreurs : {erreurs} "
                  f"| tokens : {tokens_entree} envoyés, {tokens_sortie} reçus")

if len(a_faire):
    print(f"\nTokens moyens par recette : {tokens_entree / len(a_faire):.0f} envoyés, "
          f"{tokens_sortie / len(a_faire):.0f} reçus")
print(f"Fiches disponibles dans {sortie.name} : {len(lire_fiches(sortie))}")

"""Chemins et constantes partagés par les scripts et l'application."""
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
MODELES = RACINE / "modeles"
RESULTATS = RACINE / "resultats"

CSV_BRUT = RACINE / "recipes.csv"                  # Food.com complet (700 Mo, non versionné)
ECHANTILLON = DATA / "recettes_echantillon.csv"    # notre échantillon de travail
FICHES = DATA / "fiches_llm.jsonl"                 # fiches extraites par le LLM (zéro exemple)
FICHES_FEW_SHOT = DATA / "fiches_llm_few_shot.jsonl"

PIPELINE = MODELES / "clustering.joblib"
RECETTES_TRAIN = MODELES / "recettes_train.pkl"
FAMILLES = MODELES / "familles.json"
NOMS_FAMILLES = RACINE / "noms_familles.json"      # noms donnés à la main (facultatif)

GRAINE = 42


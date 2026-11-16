"""IA prédictive, étape de préparation : transformer une fiche de recette en vecteur numérique.

Une fiche = {ingredients_principaux: [...], temps_total_minutes, cout_estime, mode_cuisson}.
Le vecteur contient :
  - temps : log(1 + minutes), standardisé. Le log, parce qu'une recette de 10 min et une de 40 min
    diffèrent autant qu'une de 1 h et une de 4 h (ce sont les rapports qui comptent, pas les écarts) ;
  - coût : 0 / 1 / 2 (faible / moyen / élevé), standardisé ;
  - mode de cuisson : 7 colonnes 0/1 (one-hot) ;
  - ingrédients : une colonne 0/1 par ingrédient du vocabulaire (multi-hot). Le vocabulaire est
    appris sur l'entraînement : ingrédients présents dans au moins `freq_min` des recettes.

Par défaut, les colonnes 0/1 ne sont PAS standardisées : un ingrédient rare (2 % des recettes) aurait
après standardisation une valeur d'environ 7 quand il est présent, et écraserait tout le reste dans le
calcul des distances. Avec `standardiser_tout=True`, on peut mesurer cet effet (voir 3_entrainer_clustering.py).
"""
from typing import get_args

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from llm import Cout, ModeCuisson

COUTS = list(get_args(Cout))
MODES = list(get_args(ModeCuisson))


def _en_dataframe(fiches):
    if isinstance(fiches, pd.DataFrame):
        return fiches
    return pd.DataFrame([f if isinstance(f, dict) else f.model_dump() for f in fiches])


class VectoriseurFiche(BaseEstimator, TransformerMixin):
    def __init__(self, freq_min=0.02, standardiser_tout=False):
        self.freq_min = freq_min
        self.standardiser_tout = standardiser_tout

    def _brut(self, df):
        temps = np.log1p(df["temps_total_minutes"].astype(float).to_numpy())[:, None]
        cout = df["cout_estime"].map(COUTS.index).to_numpy(dtype=float)[:, None]
        modes = np.array([[m == mode for mode in MODES] for m in df["mode_cuisson"]], dtype=float)
        ingr = np.array([[v in set(liste) for v in self.vocabulaire_]
                         for liste in df["ingredients_principaux"]], dtype=float)
        return np.hstack([temps, cout, modes, ingr])

    def fit(self, fiches, y=None):
        df = _en_dataframe(fiches)
        compte = pd.Series([i for liste in df["ingredients_principaux"] for i in set(liste)]).value_counts()
        self.vocabulaire_ = sorted(compte[compte >= self.freq_min * len(df)].index)
        X = self._brut(df)
        n_std = X.shape[1] if self.standardiser_tout else 2      # sinon seulement temps et coût
        self.scaler_ = StandardScaler().fit(X[:, :n_std])
        self.n_std_ = n_std
        return self

    def transform(self, fiches):
        X = self._brut(_en_dataframe(fiches))
        X[:, :self.n_std_] = self.scaler_.transform(X[:, :self.n_std_])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(["temps (log)", "coût"] + [f"cuisson : {m}" for m in MODES]
                        + [f"ingr. : {v}" for v in self.vocabulaire_])

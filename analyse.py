"""
Projet N4 - Traitement et visualisation des données
Analyse d’un jeu de données de phishing (scores d'intérêt, âge, produit recommandé, support, succès de campagne).

Ce script couvre :
1. Importation + EDA
2. Nettoyage + optimisation mémoire
3. Détection d’anomalies
4. Calcul de KPI + visualisations (matplotlib uniquement)
5. Préparation de statistiques utiles pour le data telling
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 0. Configuration de base
# -------------------------------------------------------------------
DATA_PATH = os.path.join("data", "result.csv")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)



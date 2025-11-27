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


# -------------------------------------------------------------------
# 1. Importation & EDA
# -------------------------------------------------------------------

print("=== 1. IMPORT & EDA ===")

df = pd.read_csv(DATA_PATH, sep=";")

print("\nAperçu des premières lignes :")
print(df.head())

print("\nTypes de colonnes initiaux :")
print(df.dtypes)

print("\nRésumé statistique (incluant les objets) :")
print(df.describe(include="all"))

mem_before = df.memory_usage(deep=True).sum()
print(f"\nMémoire utilisée AVANT optimisation : {mem_before/1024:.2f} Ko")

print("\nQualité initiale : valeurs manquantes & extrêmes")
print(df.isna().sum())

for col in ["gaming_interest_score",
            "insta_design_interest_score",
            "football_interest_score"]:
    print(f"{col} -> min={df[col].min()} / max={df[col].max()}")

# -------------------------------------------------------------------
# 2. Nettoyage & mise en forme
# -------------------------------------------------------------------

print("\n=== 2. NETTOYAGE & MISE EN FORME ===")

df_clean = df.copy()


df_clean["campaign_success"] = df_clean["campaign_success"].astype(str).str.strip()

df_clean["campaign_success_bool"] = df_clean["campaign_success"].map(
    {"True": 1, "False": 0}
)


df_clean["recommended_product"] = df_clean["recommended_product"].astype("string")
df_clean["recommended_product"] = df_clean["recommended_product"].replace(
    {
        "Fornite": "Fortnite",   
        "Test": pd.NA         
    }
)


df_clean["canal_recommande"] = (
    df_clean["canal_recommande"]
    .astype("string")
    .str.lower()
    .str.strip()
)


df_clean["canal_recommande"] = df_clean["canal_recommande"].replace(
    {
        "mail": "mail",
        "insta": "insta",
        "facebook": "facebook"
    }

)


print("\nValeurs manquantes AVANT traitement :")
print(df_clean.isna().sum())


df_clean = df_clean.dropna(
    subset=["football_interest_score", "recommended_product", "age"]
)

print("\nValeurs manquantes APRÈS suppression des lignes incomplètes :")
print(df_clean.isna().sum())


df_clean["Id"] = df_clean["Id"].astype("int32")
for col in [
    "gaming_interest_score",
    "insta_design_interest_score",
    "football_interest_score",
]:
    df_clean[col] = df_clean[col].astype("float32")

df_clean["age"] = df_clean["age"].astype("float32")


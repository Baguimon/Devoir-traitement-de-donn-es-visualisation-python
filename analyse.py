"""
Projet N4 - Traitement et visualisation des données
Mathis TRUONG - Timothé WINKLER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# 0. Configuration de base
DATA_PATH = os.path.join("data", "result.csv")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)


# ----------------------------
# 1. Importation & EDA

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

# ----------------------------
# 2. Nettoyage & mise en forme

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

# --------------------------
# 3. Détection d’anomalies

print("\n=== 3. DÉTECTION D’ANOMALIES ===")

for col in [
    "gaming_interest_score",
    "insta_design_interest_score",
    "football_interest_score",
]:

    df_clean[col + "_is_anomaly"] = (
        (df_clean[col] < 0) | (df_clean[col] > 100)
    )

    nb_anom = df_clean[col + "_is_anomaly"].sum()
    print(f"{col}: {nb_anom} anomalies hors [0, 100] détectées.")

    plt.figure()
    x = range(len(df_clean))
    colors = np.where(df_clean[col + "_is_anomaly"], "red", "blue")
    plt.scatter(x, df_clean[col], c=colors)
    plt.axhline(0, linestyle="--")
    plt.axhline(100, linestyle="--")
    plt.title(f"Anomalies sur {col} (rouge = anomalie)")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"anomalies_{col}.png"))
    plt.close()

anomaly_cols = [
    "gaming_interest_score_is_anomaly",
    "insta_design_interest_score_is_anomaly",
    "football_interest_score_is_anomaly",
]
mask_anomaly = df_clean[anomaly_cols].any(axis=1)
print(f"\nNombre total de lignes avec au moins une anomalie : {mask_anomaly.sum()}")

df_no_anom = df_clean.loc[~mask_anomaly].copy()
print(f"Taille du dataset après suppression des anomalies : {df_no_anom.shape}")

for col in [
    "gaming_interest_score",
    "insta_design_interest_score",
    "football_interest_score",
]:
    print(f"{col} après nettoyage -> min={df_no_anom[col].min()} / max={df_no_anom[col].max()}")


# --------------------------
# 4. Optimisation mémoire finale

print("\n=== 4. OPTIMISATION MÉMOIRE ===")

df_opt = df_no_anom.copy()

df_opt["age"] = df_opt["age"].astype("int8")

for col in [
    "recommended_product",
    "canal_recommande",
    "campaign_success",
]:
    df_opt[col] = df_opt[col].astype("category")

bins_age = [0, 12, 17, 24, 34, 44, 60, 120]
labels_age = ["0-12", "13-17", "18-24", "25-34", "35-44", "45-60", "60+"]
df_opt["age_group"] = pd.cut(
    df_opt["age"], bins=bins_age, labels=labels_age, right=True, include_lowest=True
)

bins_gaming = [-0.01, 30, 70, 100]
labels_gaming = ["faible", "moyen", "fort"]
df_opt["gaming_segment"] = pd.cut(
    df_opt["gaming_interest_score"],
    bins=bins_gaming,
    labels=labels_gaming,
    right=True,
    include_lowest=True,
)

df_opt["age_group"] = df_opt["age_group"].astype("category")
df_opt["gaming_segment"] = df_opt["gaming_segment"].astype("category")

mem_after = df_opt.memory_usage(deep=True).sum()
print(f"Mémoire utilisée APRÈS optimisation : {mem_after/1024:.2f} Ko")
print(f"Ratio mémoire après/avant : {mem_after/mem_before:.2%}")

# -------------------------------------------------------------------
# 5. KPIs & analyses statistiques

print("\n=== 5. ANALYSE STATISTIQUE & KPI ===")

def success_rate(group: pd.DataFrame) -> float:
    return group["campaign_success_bool"].mean()

global_sr = success_rate(df_opt)
print(f"Taux de réussite global de la campagne : {global_sr:.2%}")

sr_by_product = df_opt.groupby("recommended_product").apply(success_rate).sort_values(ascending=False)
print("\nTaux de réussite par produit :")
print((sr_by_product * 100).round(2).astype(str) + " %")

sr_by_channel = df_opt.groupby("canal_recommande").apply(success_rate).sort_values(ascending=False)
print("\nTaux de réussite par support :")
print((sr_by_channel * 100).round(2).astype(str) + " %")

sr_by_age_group = df_opt.groupby("age_group").apply(success_rate)
print("\nTaux de réussite par tranche d’âge :")
print((sr_by_age_group * 100).round(2).astype(str) + " %")

sr_by_gaming_seg = df_opt.groupby("gaming_segment").apply(success_rate)
print("\nTaux de réussite par segment gaming :")
print((sr_by_gaming_seg * 100).round(2).astype(str) + " %")

corr = df_opt[
    [
        "gaming_interest_score",
        "insta_design_interest_score",
        "football_interest_score",
        "age",
        "campaign_success_bool",
    ]
].corr()

print("\nMatrice de corrélation :")
print(corr)

# ----------------------------
# 5. KPIs & analyses statistiques

print("\n=== 5. ANALYSE STATISTIQUE & KPI ===")

def success_rate(group: pd.DataFrame) -> float:
    return group["campaign_success_bool"].mean()

global_sr = success_rate(df_opt)
print(f"Taux de réussite global de la campagne : {global_sr:.2%}")

sr_by_product = df_opt.groupby("recommended_product").apply(success_rate).sort_values(ascending=False)
print("\nTaux de réussite par produit :")
print((sr_by_product * 100).round(2).astype(str) + " %")

sr_by_channel = df_opt.groupby("canal_recommande").apply(success_rate).sort_values(ascending=False)
print("\nTaux de réussite par support :")
print((sr_by_channel * 100).round(2).astype(str) + " %")

sr_by_age_group = df_opt.groupby("age_group").apply(success_rate)
print("\nTaux de réussite par tranche d’âge :")
print((sr_by_age_group * 100).round(2).astype(str) + " %")

sr_by_gaming_seg = df_opt.groupby("gaming_segment").apply(success_rate)
print("\nTaux de réussite par segment gaming :")
print((sr_by_gaming_seg * 100).round(2).astype(str) + " %")

corr = df_opt[
    [
        "gaming_interest_score",
        "insta_design_interest_score",
        "football_interest_score",
        "age",
        "campaign_success_bool",
    ]
].corr()

print("\nMatrice de corrélation :")
print(corr)

# ---------------------------
# 6. Visualisations des KPI (matplotlib)

print("\n=== 6. VISUALISATIONS (figures/) ===")

plt.figure()
df_opt[["gaming_interest_score", "insta_design_interest_score", "football_interest_score"]].hist()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "distrib_scores.png"))
plt.close()

plt.figure()
sr_by_product.plot(kind="bar")
plt.ylabel("Taux de réussite")
plt.title("Taux de réussite par produit recommandé")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kpi_success_by_product.png"))
plt.close()

plt.figure()
sr_by_channel.plot(kind="bar")
plt.ylabel("Taux de réussite")
plt.title("Taux de réussite par support")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kpi_success_by_channel.png"))
plt.close()


plt.figure()
sr_by_age_group.plot(kind="bar")
plt.ylabel("Taux de réussite")
plt.title("Taux de réussite par tranche d’âge")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kpi_success_by_age_group.png"))
plt.close()

plt.figure()
plt.imshow(corr, interpolation="nearest")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "corr_matrix.png"))
plt.close()

print("Toutes les figures ont été enregistrées dans le dossier 'figures/'.")

# -------------------------
# 7. Extraction de stats pour le data telling

print("\n=== 7. EXTRACTION POUR LE DATA TELLING ===")


mask_target = (df_opt["age_group"] == "25-34") & (df_opt["gaming_segment"] == "fort")
target_segment = df_opt.loc[mask_target]

print(f"Effectif du segment cible (25-34 ans, gaming fort) : {len(target_segment)}")
print(
    "Taux de réussite dans ce segment : "
    f"{target_segment['campaign_success_bool'].mean():.2%}"
)

print("\nRépartition des produits recommandés dans ce segment :")
print(target_segment["recommended_product"].value_counts())

print("\nRépartition des supports recommandés dans ce segment :")
print(target_segment["canal_recommande"].value_counts())

print("\nScript terminé.")


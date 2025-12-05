import streamlit as st
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})

# -------------------------------------------------------------------
# TITRE + DESCRIPTION
# -------------------------------------------------------------------
st.title("Comparaison : Modèle / Observations")

st.markdown("""
L’objectif de cette application est de comparer un **modèle (CSV, 1 an)** avec des données d'observation sur **une seule année**, afin de faciliter les calculs et analyses.  

Trois types de données de référence sont disponibles :  
- **obs2000_2009** : observations 2000–2009  
- **obs** : observations 2010–2019  
- **typique** : année type  

Cet outil est principalement utilisé dans le domaine du bâtiment, notamment pour l’évaluation thermique à travers des modèles de simulation dynamique (STD).
""")

# -------------------------------------------------------------------
# PARAMÈTRES DÉJÀ EXISTANTS
# -------------------------------------------------------------------
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]

couleur_modele = "goldenrod"
couleur_TRACC = "lightgray"

vmaxT = 5
vminT = -5

vmaxP = 100
vminP = 50

vmaxH = 100
vminH = -100

vmaxDJU = 150
vminDJU = -150

# -------- Noms des mois --------
mois_noms = {
    1: "01 - Janvier",   2: "02 - Février",  3: "03 - Mars",
    4: "04 - Avril",     5: "05 - Mai",      6: "06 - Juin",
    7: "07 - Juillet",   8: "08 - Août",     9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
}

# -------------------------------------------------------------------
# 1) Construire le mapping année -> fichier
# -------------------------------------------------------------------
dossiers = ["obs2000_2009", "obs", "typique"]
annee_to_file = {}  # dict : annee -> (dossier, fichier)

for dossier in dossiers:
    for f in os.listdir(dossier):
        if f.endswith(".nc"):
            path = os.path.join(dossier, f)
            ds = xr.open_dataset(path, decode_times=True)
            if "T2m" in ds:
                years = ds["time"].dt.year.values
                for y in years:
                    annee_to_file[y] = (dossier, f)
            ds.close()

annees_dispo = sorted(annee_to_file.keys())

# -------------------------------------------------------------------
# 2) Choix de l'année
# -------------------------------------------------------------------
annee_sel = st.selectbox("Choisir l'année :", annees_dispo)

# -------------------------------------------------------------------
# 3) Trouver le fichier correspondant à l'année choisie
# -------------------------------------------------------------------
dossier_sel, fichier_sel = annee_to_file[annee_sel]
nc_path = os.path.join(dossier_sel, fichier_sel)

# -------------------------------------------------------------------
# 4) Choix de la ville (déduit automatiquement du nom du fichier)
# -------------------------------------------------------------------
ville_sel = fichier_sel.replace(".nc", "")

ds_obs = xr.open_dataset(nc_path, decode_times=True)

if "T2m" not in ds_obs:
    st.error("⚠️ La variable 'T2m' est absente du fichier NetCDF.")
    st.stop()

times = ds_obs["time"].to_series()
T = ds_obs["T2m"].to_series()

# Extraire uniquement l'année sélectionnée
mask = times.dt.year == annee_sel
obs_time = times[mask]
obs_temp = T[mask].values

# -------------------------------------------------------------------
# 5) Upload CSV du modèle (1 an = 8760 valeurs)
# -------------------------------------------------------------------
uploaded = st.file_uploader(
    "Déposer le fichier CSV du modèle (8760 valeurs horaires) :",
    type=["csv"]
)

if uploaded:
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    if len(model_values) != 8760:
        st.warning(f"⚠️ Le CSV contient {len(model_values)} valeurs, pas 8760.")

    # -------------------------------------------------------------------
    # 6) Création DataFrame OBS (1 an)
    # -------------------------------------------------------------------
    df_obs = pd.DataFrame({
        "time": obs_time,
        "T2m": obs_temp
    })

    df_obs["year"] = df_obs["time"].dt.year
    df_obs["month"] = df_obs["time"].dt.month
    df_obs["day"] = df_obs["time"].dt.day
    df_obs["month_name"] = df_obs["month"].map(mois_noms)

    st.success(f"✔ Données chargées : {ville_sel} – Année {annee_sel}")
    st.dataframe(df_obs.head())

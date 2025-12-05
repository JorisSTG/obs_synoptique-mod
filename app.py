import streamlit as st
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
- **obs** : observations 2010–2019  
- **obs2000_2009** : observations 2000–2009  
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
# 1) Sélection du type d'observation
# -------------------------------------------------------------------
type_sel = st.selectbox(
    "Choisir le type de données d'observation :",
    ["obs", "obs2000_2009", "typique"]
)

base_folder = type_sel

if not os.path.isdir(base_folder):
    st.error(f"⚠️ Le dossier {base_folder} est introuvable.")
    st.stop()

# -------------------------------------------------------------------
# 2) Liste automatique des villes disponibles
# -------------------------------------------------------------------
nc_files = [f for f in os.listdir(base_folder) if f.endswith(".nc")]

if len(nc_files) == 0:
    st.error("⚠️ Aucun fichier NetCDF trouvé dans ce dossier.")
    st.stop()

ville_list = [f.replace(".nc", "") for f in nc_files]

ville_sel = st.selectbox("Choisir la ville :", ville_list)

# -------------------------------------------------------------------
# 3) Upload CSV modèle
# -------------------------------------------------------------------
uploaded = st.file_uploader(
    "Déposer le fichier CSV du modèle (8760 valeurs horaires) :",
    type=["csv"]
)

if uploaded:

    # Lecture CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    if len(model_values) != 8760:
        st.warning(f"⚠️ Le fichier CSV contient {len(model_values)} valeurs. Une année complète = 8760 valeurs.")

    # -------------------------------------------------------------------
    # 4) Lecture du fichier NetCDF de la ville
    # -------------------------------------------------------------------
    nc_path = os.path.join(base_folder, f"{ville_sel}.nc")
    ds_obs = xr.open_dataset(nc_path, decode_times=True)

    if "T2m" not in ds_obs:
        st.error("⚠️ La variable 'T2m' est absente du fichier NetCDF.")
        st.stop()

    times = ds_obs["time"].to_series()
    T = ds_obs["T2m"].to_series()

    # -------------------------------------------------------------------
    # 5) Extraction des années disponibles
    # -------------------------------------------------------------------
    annees_dispo = sorted(times.dt.year.unique())

    if type_sel != "typique":
        annee_sel = st.selectbox("Sélectionner une année :", annees_dispo)
        mask = times.dt.year == annee_sel
        obs_time = times[mask]
        obs_temp = T[mask].values
    else:
        annee_sel = "Année typique"
        obs_time = times
        obs_temp = T.values

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

    st.success(f"✔ Données chargées : {ville_sel} – {annee_sel}")

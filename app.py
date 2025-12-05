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

import streamlit as st
import pandas as pd
import xarray as xr
import os

# Dossiers
dossiers = ["obs2000_2009", "obs", "typique"]

# Liste des villes (on suppose que toutes les villes ont le même nom de fichier dans chaque dossier)
villes = ["AGEN", "CARPENTRAS", "MACON", "MARIGNANE", "NANCY", "RENNES", "TOURS", "TRAPPES"]

# -------------------------------------------------------------------
# 1) Choix de la ville
# -------------------------------------------------------------------
ville_sel = st.selectbox("Choisir la ville :", villes)

# -------------------------------------------------------------------
# 2) Choix de l'année
# -------------------------------------------------------------------
# Exemple : on connaît les plages d'années par dossier
annees_obs2000_2009 = list(range(2000, 2010))
annees_obs2010_2019 = list(range(2010, 2020))
annees_typique = [9999]  # placeholder pour l'année typique

annees_dispo = annees_obs2000_2009 + annees_obs2010_2019 + annees_typique
annee_sel = st.selectbox("Choisir l'année :", annees_dispo)

# -------------------------------------------------------------------
# 3) Déterminer automatiquement le dossier en fonction de l'année
# -------------------------------------------------------------------
if annee_sel in annees_obs2000_2009:
    dossier_sel = "obs2000_2009"
elif annee_sel in annees_obs2010_2019:
    dossier_sel = "obs"
else:
    dossier_sel = "typique"

# -------------------------------------------------------------------
# 4) Construire le chemin vers le fichier correspondant à la ville
# -------------------------------------------------------------------
nc_file = os.path.join(dossier_sel, f"{ville_sel}.nc")
ds = xr.open_dataset(nc_file, decode_times=True)

# -------------------------------------------------------------------
# 5) Extraire uniquement les données pour l'année choisie
# -------------------------------------------------------------------
if dossier_sel != "typique":
    mask = ds["time"].dt.year == annee_sel
    obs_time = ds["time"].values[mask]
    obs_temp = ds["T2m"].values[mask]
else:
    obs_time = ds["time"].values
    obs_temp = ds["T2m"].values

# Créer le DataFrame
df_obs = pd.DataFrame({
    "time": obs_time,
    "T2m": obs_temp
})

st.write(f"Données chargées : {ville_sel} – Année {annee_sel}")
st.dataframe(df_obs.head())

import streamlit as st
import xarray as xr
import os
import pandas as pd

# Dossiers
dossiers = ["obs2000_2009", "obs", "typique"]

# Lister tous les fichiers .nc disponibles (juste le nom du fichier)
all_files = []
file_to_dossier = {}  # mapping fichier -> dossier
for dossier in dossiers:
    files = [f for f in os.listdir(dossier) if f.endswith(".nc")]
    for f in files:
        all_files.append(f)
        file_to_dossier[f] = dossier

# Menu pour choisir le fichier NetCDF (nom complet du fichier seulement)
file_sel = st.selectbox("Choisir le fichier NetCDF (ville + code) :", all_files)

# Récupérer le dossier correspondant
dossier_sel = file_to_dossier[file_sel]
nc_path = os.path.join(dossier_sel, file_sel)

# Déterminer l'année possible
if "obs2000_2009" in dossier_sel:
    annees_dispo = list(range(2000, 2010))
elif "obs" in dossier_sel:
    annees_dispo = list(range(2010, 2020))
else:
    annees_dispo = [9999]  # placeholder pour typique

# Choix de l'année
annee_sel = st.selectbox("Choisir l'année :", annees_dispo)

# Ouvrir le fichier
ds = xr.open_dataset(nc_path, decode_times=True)

# Extraire uniquement l'année sélectionnée (sauf typique)
if "typique" not in dossier_sel:
    mask = ds["time"].dt.year == annee_sel
    obs_time = ds["time"].values[mask]
    obs_temp = ds["T2"].values[mask]
else:
    obs_time = ds["time"].values
    obs_temp = ds["T2"].values

# Créer DataFrame
df_obs = pd.DataFrame({
    "time": obs_time,
    "T2m": obs_temp
})

st.write(f"Données chargées : {file_sel} – Année {annee_sel}")
st.dataframe(df_obs.head())

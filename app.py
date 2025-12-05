import streamlit as st
import xarray as xr
import os
import pandas as pd

# Dossiers
dossiers = ["obs2000_2009", "obs", "typique"]

# Lister tous les fichiers .nc disponibles
all_files = []
for dossier in dossiers:
    files = [os.path.join(dossier, f) for f in os.listdir(dossier) if f.endswith(".nc")]
    all_files.extend(files)

# Menu pour choisir le fichier complet
file_sel = st.selectbox("Choisir le fichier NetCDF (ville + code) :", all_files)

# Déterminer l'année possible
if "obs2000_2009" in file_sel:
    annees_dispo = list(range(2000, 2010))
elif "obs" in file_sel:
    annees_dispo = list(range(2010, 2020))
else:
    annees_dispo = [9999]  # placeholder pour typique

# Choix de l'année
annee_sel = st.selectbox("Choisir l'année :", annees_dispo)

# Ouvrir le fichier
ds = xr.open_dataset(file_sel, decode_times=True)

# Extraire uniquement l'année sélectionnée (sauf typique)
if "typique" not in file_sel:
    mask = ds["time"].dt.year == annee_sel
    obs_time = ds["time"].values[mask]
    obs_temp = ds["T2m"].values[mask]
else:
    obs_time = ds["time"].values
    obs_temp = ds["T2m"].values

# Créer DataFrame
df_obs = pd.DataFrame({
    "time": obs_time,
    "T2m": obs_temp
})

st.write(f"Données chargées : {file_sel} – Année {annee_sel}")
st.dataframe(df_obs.head())

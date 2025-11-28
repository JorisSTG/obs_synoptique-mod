#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 14:23:23 2025

@author: saint-genesj
"""

import streamlit as st
import xarray as xr
import pandas as pd
import os
import io

st.title("Comparateur Observations / Modèle (NetCDF)")

st.write("Ce site compare un fichier modèle NetCDF avec les observations disponibles.")

# --------- CHARGEMENT DES OBSERVATIONS ---------

@st.cache_data
def load_all_obs():
    obs_dir = "obs"
    obs_files = [f for f in os.listdir(obs_dir) if f.endswith(".nc")]

    datasets = []
    for f in obs_files:
        ds = xr.open_dataset(os.path.join(obs_dir, f))
        datasets.append(ds)

    # concaténation sur le temps
    obs_all = xr.concat(datasets, dim="time")
    return obs_all

st.write("Chargement des observations…")
obs = load_all_obs()
st.success(f"Observations chargées : {len(obs.time)} pas de temps")

# --------- UPLOAD DU MODELE ---------

uploaded = st.file_uploader("Dépose ton fichier modèle (.nc)", type=["nc"])

if uploaded:
    # lecture du fichier envoyé
    model_ds = xr.open_dataset(uploaded)

    st.write("### Exemple du contenu du modèle")
    st.write(model_ds)

    # Vérification
    if "T" not in model_ds:
        st.error("La variable 'T' est absente dans le modèle.")
    else:
        st.success("La variable T a été trouvée dans le modèle ✔")

        # --- COMPARAISON ---

        # rééchantillonner si nécessaire
        common_times = np.intersect1d(obs.time.values, model_ds.time.values)

        if len(common_times) == 0:
            st.error("Aucun pas de temps en commun entre observations et modèle.")
        else:
            st.success(f"{len(common_times)} pas de temps communs trouvés.")

            obs_c = obs.sel(time=common_times)
            mod_c = model_ds.sel(time=common_times)

            comparaison = obs_c["T"] - mod_c["T"]

            st.write("### Exemple comparaison (obs - modèle)")
            st.write(comparaison.isel(time=0))

            # export NetCDF
            output = io.BytesIO()
            comparaison.to_netcdf(output)
            output.seek(0)

            st.download_button(
                "Télécharger le fichier NetCDF des erreurs",
                output,
                file_name="comparaison_obs_modele.nc",
                mime="application/x-netcdf"
            )
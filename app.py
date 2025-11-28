import streamlit as st
import pandas as pd
import xarray as xr
import glob
import os
import numpy as np
import io

st.title("Comparaison modèle CSV / Observations NetCDF")

# -------- Paramètres --------
base_folder = "obs"  # dossier contenant les fichiers NetCDF
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]  # année non bissextile

# -------- Liste des fichiers NetCDF --------
nc_files = glob.glob(os.path.join(base_folder, "*.nc"))
# Pour simplifier, on affiche le nom du fichier complet dans la liste déroulante
ville_sel = st.selectbox("Choisir le fichier NetCDF :", [os.path.basename(f) for f in nc_files])
nc_file_sel = os.path.join(base_folder, ville_sel)

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Dépose ton fichier CSV modèle (colonne unique T) :", type=["csv"])

# -------- Seuils --------
tmin_threshold = st.number_input("Seuil Tmin pour compter les heures supérieures (°C)", value=20)
tmax_thresholds = st.text_input("Seuils Tmax pour compter les heures supérieures (°C, séparés par des virgules)", "25,30,35")

if uploaded:
    # Lecture du CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:,0].values

    # Lecture NetCDF sélectionné
    ds_obs = xr.open_dataset(nc_file_sel)
    if "T2m" in ds_obs:
        obs_values = ds_obs["T2m"].values.flatten()
    elif "T" in ds_obs:
        obs_values = ds_obs["T"].values.flatten()
    else:
        st.error("Le NetCDF n'a pas de variable T ou T2m")
        st.stop()

    # Vérification longueur
    if len(model_values) != len(obs_values):
        st.warning(f"⚠ Longueur différente modèle / obs : {len(model_values)} vs {len(obs_values)}")
        min_len = min(len(model_values), len(obs_values))
        model_values = model_values[:min_len]
        obs_values = obs_values[:min_len]

    # -------- RMSE mensuel --------
    def rmse(a,b):
        return np.sqrt(np.mean((a-b)**2))

    start_idx = 0
    resultats = []
    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        end_idx = start_idx + nb_heures
        mod_mois = model_values[start_idx:end_idx]
        obs_mois = obs_values[start_idx:end_idx]

        val_rmse = rmse(obs_mois, mod_mois)
        resultats.append({
            "Fichier_NetCDF": ville_sel,
            "Mois": mois,
            "RMSE": val_rmse
        })
        start_idx = end_idx

    df_rmse = pd.DataFrame(resultats)
    st.subheader("RMSE mensuel")
    st.dataframe(df_rmse)

    # -------- Nombre d'heures au-dessus ou en dessous des seuils --------
    tmax_thresholds_list = [float(x.strip()) for x in tmax_thresholds.split(",")]

    df_stats = pd.DataFrame({
        "Seuil": [],
        "Nb_heures_sup": [],
        "Nb_heures_inf": []
    })

    for seuil in tmax_thresholds_list:
        nb_sup = np.sum(model_values > seuil)
        nb_inf = np.sum(model_values <= seuil)
        df_stats = pd.concat([df_stats, pd.DataFrame([{"Seuil": seuil, "Nb_heures_sup": nb_sup, "Nb_heures_inf": nb_inf}])], ignore_index=True)

    st.subheader("Nombre d'heures au-dessus ou en dessous des seuils")
    st.dataframe(df_stats)

    # -------- Export CSV --------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_rmse.to_excel(writer, sheet_name="RMSE_mensuel", index=False)
        df_stats.to_excel(writer, sheet_name="Stats_seuils", index=False)
    output.seek(0)

    st.download_button(
        label="Télécharger les résultats en Excel",
        data=output,
        file_name=f"resultats_{ville_sel}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

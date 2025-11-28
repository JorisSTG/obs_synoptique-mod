import streamlit as st
import pandas as pd
import xarray as xr
import glob
import os
import numpy as np
import io

st.title("Comparaison modèle CSV / Observations NetCDF sur 10 ans")

# -------- Paramètres --------
base_folder = "obs"  # dossier contenant les fichiers NetCDF
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]  # année non bissextile

# -------- Liste des fichiers NetCDF --------
nc_files = glob.glob(os.path.join(base_folder, "*.nc"))
if not nc_files:
    st.error(f"Aucun fichier NetCDF trouvé dans {base_folder}")
    st.stop()

ville_sel = st.selectbox("Choisir le fichier NetCDF :", [os.path.basename(f) for f in nc_files])
nc_file_sel = os.path.join(base_folder, ville_sel)

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Dépose ton fichier CSV modèle (colonne unique T) :", type=["csv"])

# -------- Seuils --------
t_thresholds = st.text_input("Seuils Tmax pour compter les heures supérieures (°C, séparés par des virgules)", "25,30,35")

if uploaded:
    # Lecture du CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    # Lecture NetCDF sélectionné avec décodage du temps
    ds_obs = xr.open_dataset(nc_file_sel, decode_times=True)
    if "T" not in ds_obs:
        st.error("Le NetCDF n'a pas de variable 'T'")
        st.stop()

    # Conversion en DataFrame avec datetime
    obs_series = ds_obs["T"].to_series()  # index = datetime
    df_obs = obs_series.reset_index()
    df_obs.rename(columns={"T": "T", "time": "time"}, inplace=True)
    df_obs["year"] = df_obs["time"].dt.year
    df_obs["month"] = df_obs["time"].dt.month
    df_obs["day"] = df_obs["time"].dt.day

    # -------- Supprimer les 29 février --------
    df_obs = df_obs[~((df_obs["month"] == 2) & (df_obs["day"] == 29))]

    # -------- RMSE mensuel --------
    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    df_rmse = []
    start_idx_model = 0
    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        # Extraire les valeurs modèle pour ce mois
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        mod_sorted = np.sort(mod_mois)

        # Extraire toutes les valeurs observées du mois sur 10 ans
        obs_mois_10ans = []
        for year in sorted(df_obs["year"].unique()):
            obs_year_mois = df_obs[(df_obs["year"] == year) & (df_obs["month"] == mois)]["T"].values
            obs_mois_10ans.append(np.sort(obs_year_mois))

        # Moyenne sur 10 ans position par position
        obs_moyenne_tri = np.mean(obs_mois_10ans, axis=0)

        val_rmse = rmse(mod_sorted, obs_moyenne_tri)
        df_rmse.append({"Fichier_NetCDF": ville_sel, "Mois": mois, "RMSE": val_rmse})

        start_idx_model += nb_heures

    df_rmse = pd.DataFrame(df_rmse)
    st.subheader("RMSE mensuel")
    st.dataframe(df_rmse)

    # -------- Nombre moyen d'heures au-dessus d'un seuil --------
    t_thresholds_list = [float(x.strip()) for x in t_thresholds.split(",")]
    df_stats = []
    for seuil in t_thresholds_list:
        for mois in range(1, 13):
            heures_par_mois = []
            for year in sorted(df_obs["year"].unique()):
                obs_year_mois = df_obs[(df_obs["year"] == year) & (df_obs["month"] == mois)]["T"].values
                heures_par_mois.append(np.sum(obs_year_mois > seuil))
            nb_heures_moy = np.mean(heures_par_mois)
            df_stats.append({"Fichier_NetCDF": ville_sel, "Mois": mois, "Seuil": seuil, "Nb_heures_moy": nb_heures_moy})

    df_stats = pd.DataFrame(df_stats)
    st.subheader("Nombre moyen d'heures au-dessus des seuils par mois")
    st.dataframe(df_stats)

    # -------- Export Excel --------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_rmse.to_excel(writer, sheet_name="RMSE_mensuel", index=False)
        df_stats.to_excel(writer, sheet_name="Heures_seuils", index=False)
    output.seek(0)

    st.download_button(
        label="Télécharger les résultats en Excel",
        data=output,
        file_name=f"resultats_{ville_sel}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

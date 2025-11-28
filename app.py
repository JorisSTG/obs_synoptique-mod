import streamlit as st
import pandas as pd
import xarray as xr
import glob
import os
import numpy as np

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
    obs_series = ds_obs["T"].to_series()
    df_obs = obs_series.reset_index()
    df_obs.rename(columns={"T": "T", "time": "time"}, inplace=True)
    df_obs["year"] = df_obs["time"].dt.year
    df_obs["month"] = df_obs["time"].dt.month
    df_obs["day"] = df_obs["time"].dt.day

    # Supprimer les 29 février
    df_obs = df_obs[~((df_obs["month"] == 2) & (df_obs["day"] == 29))]

    # -------- Calcul percentiles et RMSE --------
    def rmse(a, b):
        return np.sqrt(np.nanmean((a - b) ** 2))

    results = []

    start_idx_model = 0
    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        # Extraction des valeurs modèle pour ce mois
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        mod_sorted = np.sort(mod_mois)

        # Extraction des valeurs observées pour ce mois sur 10 ans
        obs_mois_10ans = []
        for year in sorted(df_obs["year"].unique()):
            vals = df_obs[(df_obs["year"] == year) & (df_obs["month"] == mois)]["T"].values
            obs_mois_10ans.append(np.sort(vals))
        
        # Alignement sur la longueur minimale
        min_len = min(len(mod_sorted), min(len(arr) for arr in obs_mois_10ans))
        obs_mois_trimmed = np.array([arr[:min_len] for arr in obs_mois_10ans])
        obs_moyenne = np.mean(obs_mois_trimmed, axis=0)

        val_rmse = rmse(mod_sorted[:min_len], obs_moyenne)
        results.append({"Mois": mois, "RMSE_percentiles": val_rmse})

        start_idx_model += nb_heures

    df_rmse = pd.DataFrame(results)
    st.subheader("RMSE sur les percentiles mensuels")
    st.dataframe(df_rmse)

    # -------- Nombre moyen d'heures au-dessus d'un seuil --------
    t_thresholds_list = [float(x.strip()) for x in t_thresholds.split(",")]
    stats = []

    for seuil in t_thresholds_list:
        for mois in range(1, 13):
            heures_mois = []
            for year in sorted(df_obs["year"].unique()):
                vals = df_obs[(df_obs["year"] == year) & (df_obs["month"] == mois)]["T"].values
                heures_mois.append(np.sum(vals > seuil))
            stats.append({"Mois": mois, "Seuil": seuil, "Nb_heures_moy": np.mean(heures_mois)})

    df_stats = pd.DataFrame(stats)
    st.subheader("Nombre moyen d'heures au-dessus des seuils")
    st.dataframe(df_stats)

    # -------- Export CSV --------
    df_rmse.to_csv("RMSE_percentiles.csv", index=False)
    df_stats.to_csv("Heures_au_dessus_seuils.csv", index=False)

    st.download_button("Télécharger RMSE", "RMSE_percentiles.csv", "text/csv")
    st.download_button("Télécharger stats heures", "Heures_au_dessus_seuils.csv", "text/csv")

    # -------- Graphique CDF --------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    for mois in [1,2,3,4,5,6,7,8,9,10,11,12]:  # exemple: janvier, juillet, décembre
        obs_concat = np.concatenate([arr for arr in obs_mois_10ans])
        obs_sorted = np.sort(obs_concat)
        mod_sorted_month = np.sort(model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])])
        plt.plot(obs_sorted, np.linspace(0,1,len(obs_sorted)), label=f"Obs mois {mois}")
        plt.plot(mod_sorted_month, np.linspace(0,1,len(mod_sorted_month)), '--', label=f"Modèle mois {mois}")
    plt.xlabel("Température")
    plt.ylabel("CDF")
    plt.legend()
    st.pyplot(plt)

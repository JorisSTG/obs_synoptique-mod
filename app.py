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
t_sup_thresholds = st.text_input("Seuils Tmax sup (°C, séparés par des virgules)", "25,30,35")
t_inf_thresholds = st.text_input("Seuils Tmin inf (°C, séparés par des virgules)", "0,5,10")

if uploaded:
    # Lecture du CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    # Lecture NetCDF sélectionné
    ds_obs = xr.open_dataset(nc_file_sel, decode_times=True)
    if "T" not in ds_obs:
        st.error("Le NetCDF n'a pas de variable 'T'")
        st.stop()

    # Conversion en DataFrame
    obs_series = ds_obs["T"].to_series()
    df_obs = obs_series.reset_index()
    df_obs.rename(columns={"T": "T", "time": "time"}, inplace=True)
    df_obs["year"] = df_obs["time"].dt.year
    df_obs["month"] = df_obs["time"].dt.month
    df_obs["day"] = df_obs["time"].dt.day

    # Supprimer les 29 février
    df_obs = df_obs[~((df_obs["month"] == 2) & (df_obs["day"] == 29))]

    # -------- RMSE sur percentiles --------
    def rmse(a, b):
        return np.sqrt(np.nanmean((a - b) ** 2))

    results_rmse = []
    start_idx_model = 0
    obs_mois_all = []

    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        # Modèle pour ce mois
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        mod_sorted = np.sort(mod_mois)

        # Observations pour ce mois sur 10 ans
        obs_mois_10ans = []
        for year in sorted(df_obs["year"].unique()):
            vals = df_obs[(df_obs["year"] == year) & (df_obs["month"] == mois)]["T"].values
            obs_mois_10ans.append(np.sort(vals))

        # Moyenne sur 10 ans
        min_len = min(len(mod_sorted), min(len(arr) for arr in obs_mois_10ans))
        obs_mois_trimmed = np.array([arr[:min_len] for arr in obs_mois_10ans])
        obs_moyenne = np.mean(obs_mois_trimmed, axis=0)

        val_rmse = rmse(mod_sorted[:min_len], obs_moyenne)
        results_rmse.append({"Mois": mois, "RMSE_percentiles": val_rmse})

        obs_mois_all.append(obs_mois_trimmed)
        start_idx_model += nb_heures

    df_rmse = pd.DataFrame(results_rmse)
    st.subheader("RMSE sur les percentiles mensuels")
    st.dataframe(df_rmse)

    # -------- Nombre moyen d'heures au-dessus / en dessous des seuils --------
    t_sup_thresholds_list = [float(x.strip()) for x in t_sup_thresholds.split(",")]
    t_inf_thresholds_list = [float(x.strip()) for x in t_inf_thresholds.split(",")]

    stats = []

    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
        obs_mois_10ans = obs_mois_all[mois-1]

        # Heures supérieures
        for seuil in t_sup_thresholds_list:
            heures_obs = [np.sum(arr > seuil)/10 for arr in obs_mois_10ans]  # moyenne par an
            nb_heures_obs_moy = np.mean(heures_obs)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = nb_heures_obs_moy - nb_heures_mod
            stats.append({
                "Mois": mois,
                "Seuil": seuil,
                "Type": "Supérieur",
                "Nb_heures_obs_moy": nb_heures_obs_moy,
                "Nb_heures_mod": nb_heures_mod,
                "Ecart_obs_mod": ecart
            })

        # Heures inférieures
        for seuil in t_inf_thresholds_list:
            heures_obs = [np.sum(arr < seuil)/10 for arr in obs_mois_10ans]  # moyenne par an
            nb_heures_obs_moy = np.mean(heures_obs)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = nb_heures_obs_moy - nb_heures_mod
            stats.append({
                "Mois": mois,
                "Seuil": seuil,
                "Type": "Inférieur",
                "Nb_heures_obs_moy": nb_heures_obs_moy,
                "Nb_heures_mod": nb_heures_mod,
                "Ecart_obs_mod": ecart
            })

    df_stats = pd.DataFrame(stats)
    st.subheader("Nombre moyen d'heures par seuil et écart obs-mod")
    st.dataframe(df_stats)

    # -------- Export CSV --------
    df_rmse.to_csv("RMSE_percentiles.csv", index=False)
    df_stats.to_csv("Heures_seuils.csv", index=False)

    st.download_button("Télécharger RMSE", "RMSE_percentiles.csv", "text/csv")
    st.download_button("Télécharger stats heures", "Heures_seuils.csv", "text/csv")

    # -------- Graphiques CDF pour chaque mois --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")

    for mois in range(1, 13):
        obs_mois_concat = np.concatenate(obs_mois_all[mois-1])
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
        df_cdf = pd.DataFrame({
            "Obs": np.sort(obs_mois_concat),
            "Mod": np.sort(mod_mois)
        })
        # On normalise pour obtenir CDF
        df_cdf["Obs_CDF"] = np.linspace(0, 1, len(df_cdf))
        df_cdf["Mod_CDF"] = np.linspace(0, 1, len(df_cdf))
        st.write(f"Mois {mois}")
        st.line_chart(df_cdf[["Obs_CDF", "Mod_CDF"]])

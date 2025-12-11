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

st.title("Comparaison : Modèle / Observations")
st.markdown(
    """
    L’objectif de cette application est d’évaluer la précision de données météorologiques en les comparant à des données de référence.
    La température de l'air au niveau du sol est l'unique paramètre utilisé afin de comparer le **modèle**.
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]
couleur_modele = "goldenrod"
couleur_Observations = "lightgray"
vmaxT = 5
vminT = -5
vmaxP = 100
vminP = 50
vmaxH = 100
vminH = -100
vmaxJ = 10
vminJ = -10
vmaxDJU = 150
vminDJU = -150

# -------- Noms des mois --------
mois_noms = {
    1: "01 - Janvier", 2: "02 - Février", 3: "03 - Mars",
    4: "04 - Avril", 5: "05 - Mai", 6: "06 - Juin",
    7: "07 - Juillet", 8: "08 - Août", 9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
}

# -------- Dossiers et années disponibles --------
dossiers = {
    "Typique": ["typique"],
    "Typique chaude": ["typique_VC"],
    "Typique froide": ["typique_VF"],
    "2000-2009": list(range(2000, 2010)),
    "2010-2019": list(range(2010, 2020))
}

# -------- Sélection de l'année ou "Typique" --------
annees_dispo = ["Typique"] + ["Typique chaude"] + ["Typique froide"]  + list(range(2000, 2020))
annee_sel = st.selectbox("Choisir l'année ou 'Typique (VC/Normal/VF)' :", annees_dispo)

# -------- Déterminer le dossier et lister les fichiers --------
if annee_sel == "Typique":
    dossier_sel = "typique"
    annee_sel = 9999  # Placeholder pour typique
elif annee_sel == "Typique chaude":
    dossier_sel = "typique_VC"
    annee_sel = 9999  # Placeholder pour typique
elif annee_sel == "Typique froide":
    dossier_sel = "typique_VF"
    annee_sel = 9999  # Placeholder pour typique
elif annee_sel in range(2000, 2010):
    dossier_sel = "obs_2000_2009"
elif annee_sel in range(2010, 2020):
    dossier_sel = "obs_2010_2019"
else:
    st.error("Sélection invalide.")
    st.stop()

# -------- Lister les fichiers .nc disponibles --------
all_files = [f for f in os.listdir(dossier_sel) if f.endswith(".nc")]

if not all_files:
    st.error(f"Aucun fichier .nc trouvé dans le dossier {dossier_sel}.")
    st.stop()

# -------- Sélection du fichier NetCDF --------
file_sel = st.selectbox("Choisir le fichier NetCDF :", all_files)

# -------- Chemin du fichier --------
nc_path = os.path.join(dossier_sel, file_sel)

# -------- Ouvrir le fichier NetCDF --------
ds = xr.open_dataset(nc_path, decode_times=True)

# -------- Vérification du contenu du fichier --------
if "time" not in ds or "T" not in ds:
    st.error("Le fichier sélectionné ne contient pas les variables 'time' et 'T'.")
    st.stop()

# -------- Extraction brut des dates et températures --------
# Conversion en DateTimeIndex → indispensable pour manipuler facilement les dates
time_all = ds["time"].to_index()      # pandas DateTimeIndex
temp_all = ds["T"].values             # tableau numpy

# -------- Suppression des 29 février (toujours absent dans le modèle CSV) --------
mask_no_feb29 = ~((time_all.month == 2) & (time_all.day == 29))
time_all = time_all[mask_no_feb29]
temp_all = temp_all[mask_no_feb29]

# -------- Sélection des données selon l'année choisie --------
if annee_sel != 9999:   # cas normal (pas "Typique")
    mask_year = (time_all.year == annee_sel)
    obs_time = time_all[mask_year]
    obs_temp = temp_all[mask_year]

    # Vérification si l'année existe dans les données NetCDF
    if len(obs_temp) == 0:
        st.error(f"L'année {annee_sel} n'est pas disponible dans les données du fichier {file_sel}.")
        st.stop()
else:
    # Mode "Typique" → aucune sélection d'année
    obs_time = time_all
    obs_temp = temp_all

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Déposer le fichier CSV du modèle (colonne unique T°C) :", type=["csv"])

if uploaded:
    st.markdown("")

    # -------- Lecture CSV modèle --------
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    def count_hours_in_bins(temp_hourly, bins):
        counts, _ = np.histogram(temp_hourly, bins=bins)
        return counts

    # -------- RMSE --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))

    def rmse_hours(obs_counts, mod_counts):
        min_len = min(len(obs_counts), len(mod_counts))
        return np.sqrt(np.nanmean((np.array(obs_counts[:min_len]) - np.array(mod_counts[:min_len]))**2))

    # -------- Nouvelle fonction : indice de recouvrement --------
    def precision_overlap(a, b, bin_width=1.0):
        """
        Calcule l'indice de recouvrement (%) entre deux séries de données.
        bin_width : largeur des tranches pour l'histogramme (en °C)
        """
        if len(a) == 0 or len(b) == 0:
            return np.nan
    
        # Définir les bornes de l'histogramme
        min_val = min(np.min(a), np.min(b))
        max_val = max(np.max(a), np.max(b))
        bins = np.arange(min_val, max_val + bin_width, bin_width)
    
        # Calcul des histogrammes normalisés
        hist_a, _ = np.histogram(a, bins=bins, density=True)
        hist_b, _ = np.histogram(b, bins=bins, density=True)
    
        # Indice de recouvrement
        overlap = np.sum(np.minimum(hist_a, hist_b) * bin_width)
        indice_percent = overlap * 100
        return round(indice_percent, 2)
    
    # -------- Boucle sur les mois --------
    results_rmse = []
    obs_mois_all = []
    start_idx_model = 0  # utile uniquement pour découper le modèle
    
    # Bins fixes pour tous les mois → meilleure cohérence RMSE_hours
    bins = np.arange(-30, 60 + 1, 1)  # de -30°C à +60°C, pas de 1°C
    
    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
    
        mois = mois_noms[mois_num]
    
        # -------- Observations : sélection horaire propre --------
        mask_mois = (obs_time.month == mois_num)
        obs_mois_vals = obs_temp[mask_mois]  # toutes les heures du mois
        obs_mois_all.append(obs_mois_vals)
    
        # -------- Modèle : découpe par bloc d'heures --------
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
    
        # -------- RMSE classique --------
        val_rmse = rmse(mod_mois, obs_mois_vals)
    
        # -------- Comptage horaire par intervalle --------
        obs_counts = count_hours_in_bins(obs_mois_vals, bins)
        mod_counts = count_hours_in_bins(mod_mois, bins)
    
        # -------- RMSE_hours --------
        val_rmse_h = rmse_hours(obs_counts, mod_counts)
    
        # -------- Indice de recouvrement (distribution entière) --------
        pct_precision = precision_overlap(mod_mois, obs_mois_vals)
    
        # -------- Stockage --------
        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "RMSE (heures)": round(val_rmse_h, 2),
            "Précision (%)": pct_precision
        })
    
        # -------- Avancer dans le modèle --------
        start_idx_model += nb_heures
    
    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    
    df_rmse_styled = (
        df_rmse.style
        .background_gradient(subset=["Précision (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None)
        .format({"Précision (%)": "{:.2f}", "RMSE (°C)": "{:.2f}", "RMSE (heures)": "{:.2f}"})
    )

    st.subheader("Précision du modèle : RMSE (°C), RMSE (heures) et précision (%)")
    st.markdown(
        """
        **Percentiles** : Une des 99 valeurs qui divisent les données de la TRACC ou du modèle 
        en 100 parts égales. Caractérise la distribution des valeurs
        
        **Exemple** : Le P95 correspond à la température pour laquelle 95% pour des valeurs sont inférieures
        
        <div style="text-align: center ;">
        <a href="https://www.allaboutlean.com/wp-content/uploads/2020/01/Normal-Distribution-and-Percentiles.png"
        target="_blank">
        <img
        src="https://www.allaboutlean.com/wp-content/uploads/2020/01/Normal-Distribution-and-Percentiles.png" width="500">
        <a/>
        </div>
        
        **RMSE** : Sert à quantifier les différences de températures
        qu'il existe entre les percentiles issues du modèle et celles de la TRACC

        **Precision** : La méthode de la **distance de Bhattacharyya** est utilisée afin de comparer 
        la distribution de deux distributions de probabilités discrètes
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df_rmse_styled, hide_index=True)


    # -------- Précision globale annuelle --------
    model_annee = model_values[:sum(heures_par_mois)]        # toutes les heures de l'année
    obs_annee = np.concatenate(obs_mois_all)                # toutes les heures TRACC concaténées
    
    precision_annuelle = precision_overlap(model_annee, obs_annee)
    st.subheader(f"Précision globale annuelle : {precision_annuelle} %")

    st.markdown(
        """
        - La précision correspond à la proportion de surface commune entre les deux histogrammes horaires (Modèle et observation) calculé selon l'indicde de **Bhattacharyya**. La valeur est alors comprise entre 0 et 100%
        - **Exemple** : Les schémas ci-dessous servent d'exemple. La valeur de la précision correspond à l'air bleu foncé vis à vis de l'air normalisé des histogrammes
        
        <div style="text-align: center ;">
        <a href="https://www.researchgate.net/profile/Md-Haidar-Sharif/publication/220804172/figure/fig3/AS:295583899242509@1447484101456/Bhattacharyya-distance-surrounds-altogether-for-one-dimensional-example-of-twosomes-of.png"
        target="_blank">
        <img
        src="https://www.researchgate.net/profile/Md-Haidar-Sharif/publication/220804172/figure/fig3/AS:295583899242509@1447484101456/Bhattacharyya-distance-surrounds-altogether-for-one-dimensional-example-of-twosomes-of.png" width="800">
        <a/>
        </div>

        """,
        unsafe_allow_html=True
    )
   
    st.subheader("")

    # -------- Seuils --------
    t_sup_thresholds = st.text_input("Seuils supérieurs (°C, séparer les seuils par des / )", "25/30/35")
    t_inf_thresholds = st.text_input("Seuils inférieurs (°C, séparer les seuils par des / )", "-5/0/5")
    t_sup_thresholds_list = [int(float(x.strip())) for x in t_sup_thresholds.split("/")]
    t_inf_thresholds_list = [int(float(x.strip())) for x in t_inf_thresholds.split("/")]
    
    stats_sup = []
    stats_inf = []
    
    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_mois = model_values[idx0:idx1]
        obs_mois = obs_mois_all[mois_num-1]
    
        # Seuils supérieurs
        for seuil in t_sup_thresholds_list:
            heures_obs = np.sum(obs_mois > seuil)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - Observations
            stats_sup.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures Observations": heures_obs,
                "Ecart (Modèle - Observations)": ecart
            })
        
        # Seuils inférieurs
        for seuil in t_inf_thresholds_list:
            heures_obs = np.sum(obs_mois < seuil)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - Observations
            stats_inf.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures Observations": heures_obs,
                "Ecart (Modèle - Observations)": ecart
            })
    
    # Création des DataFrames
    df_sup = pd.DataFrame(stats_sup)
    df_inf = pd.DataFrame(stats_inf)
    
    # Conversion en int
    for df in [df_sup, df_inf]:
        df["Heures Modèle"] = df["Heures Modèle"].astype(int)
        df["Heures Observations"] = df["Heures Observations"].astype(int)
        df["Ecart (Modèle - Observations)"] = df["Ecart (Modèle - Observations)"].astype(int)
    
    # Style : seuils supérieurs → rouge = plus chaud
    df_sup_styled = (
        df_sup.style
        .background_gradient(subset=["Ecart (Modèle - Observations)"], cmap="bwr", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures supérieur au(x) seuil(s)")
    st.dataframe(df_sup_styled, hide_index=True)
    
    # Style : seuils inférieurs → rouge = plus froid
    # Pour inverser les couleurs, on peut juste inverser le cmap
    df_inf_styled = (
        df_inf.style
        .background_gradient(subset=["Ecart (Modèle - Observations)"], cmap="bwr_r", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures inférieur au(x) seuil(s)")
    st.dataframe(df_inf_styled, hide_index=True)


    # -------- Histogrammes par plage de température --------
    st.subheader(f"Histogrammes horaire : Modèle et observations {file_sel}")
    st.markdown(
        """
        La valeur de chaque barre est égale au total d'heure compris entre [ X°C , X+1°C [
        """,
        unsafe_allow_html=True
    )
    # Bins correspondant à [X, X+1[ pour chaque température entière
    bin_edges = bins = np.arange(-5, 46, 1)  # bornes des bins
    bin_labels = bin_edges[:-1].astype(int)  # labels = début de l'intervalle
    
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        
        # Observations
        obs_hourly = obs_mois_all[mois_num-1]
        obs_counts = count_hours_in_bins(obs_hourly, bin_edges)
        
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        mod_counts = count_hours_in_bins(mod_hourly, bin_edges)
        
        # Préparer le DataFrame pour le plot
        df_plot = pd.DataFrame({
            "Temp_Num": bin_labels,
            "Température": bin_labels.astype(str),
            "Observations": obs_counts,
            "Modèle": mod_counts
        }).sort_values("Temp_Num")
        
        # Création du plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df_plot["Temp_Num"] - 0.25, df_plot["Observations"], width=0.4, label=f"Observations {file_sel}", color=couleur_Observations)
        ax.bar(df_plot["Temp_Num"] + 0.25, df_plot["Modèle"], width=0.4, label="Modèle", color=couleur_modele)
        ax.set_title(f"{mois} - Durée en heure par seuil de température")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Durée en heure")
        ax.legend(fontsize='large')
        st.pyplot(fig)
        plt.close(fig)

    # -------- Histogramme annuel par plage de température --------
    st.subheader(f"Histogramme annuel : Modèle et Observations {file_sel}")
    st.markdown(
        """
        La valeur de chaque barre est égale au total d'heures compris entre [ X°C , X+1°C [
        sur l'année entière.
        """,
        unsafe_allow_html=True
    )
    
    # Bins correspondant à [X, X+1[
    bin_edges = np.arange(-5, 46, 1)
    bin_labels = bin_edges[:-1].astype(int)
    
    # -------- Regroupement ANNUEL --------
    # Observations : concaténer tous les mois
    obs_hourly_annual = np.concatenate(obs_mois_all)
    
    # Modèle : toutes les valeurs de l'année
    mod_hourly_annual = model_values  # déjà une série horaire complète
    
    # Comptages annuels
    obs_counts_annual = count_hours_in_bins(obs_hourly_annual, bin_edges)
    mod_counts_annual = count_hours_in_bins(mod_hourly_annual, bin_edges)

    diff_counts_annual_Observations = np.maximum(0, obs_counts_annual - mod_counts_annual)
    diff_counts_annual_modele = np.maximum(0, mod_counts_annual - obs_counts_annual)

    # Préparer DataFrame pour le plot
    df_plot_year = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        "Observations": obs_counts_annual,
        "Modèle": mod_counts_annual
    }).sort_values("Temp_Num")
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year["Temp_Num"] - 0.25, df_plot_year["Observations"], width=0.4,
           label=f"Observations {file_sel}", color=couleur_Observations)
    ax.bar(df_plot_year["Temp_Num"] + 0.25, df_plot_year["Modèle"], width=0.4,
           label="Modèle", color=couleur_modele)

    fig_hist_year = fig
    ax.set_title("Année entière - Durée en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend(fontsize='large')
    
    st.pyplot(fig)
    plt.close(fig)

    # Préparer DataFrame pour le plot
    df_plot_year = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        "Différence absolue modele": diff_counts_annual_modele,
        "Différence absolue Observations": diff_counts_annual_Observations
    }).sort_values("Temp_Num")
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year["Temp_Num"], df_plot_year["Différence absolue modele"], width=0.8,
           label="Différence : Modèle > Observations", color=couleur_modele)
    
    ax.bar(df_plot_year["Temp_Num"], df_plot_year["Différence absolue Observations"], width=0.8,
           label="Différence : Modèle < Observations", color=couleur_Observations)
    
    ax.set_title("Année entière - Différence en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend(fontsize='large')
    fig_hist_diff = fig
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        La couleur de la différence est définie ainsi :

        Barres jaunes : le modèle compte davantage d’heures que les données d'observations dans cette plage de température.

        Barres blanches : les données d'boservations compte davantage d’heures que le modèle dans cette plage de température.

        La conclusion dépend donc de l’endroit où se situe cette différence. Une analyse doit être réalisée manuellement : par exemple, si l'observation présente plus d’heures dans les plages « froides », cela signifie qu’elle est globalement plus froide que le modèle.
        Comme les deux séries possèdent le même nombre total d’heures, un excès d’heures froides dans la série d'bservations implique mécaniquement un excès d’heures chaudes dans le modèle (et inversement).
        """,
        unsafe_allow_html=True
    )

    # =============================
    # Comparaison annuelle histogrammes horaires
    # =============================
    
    # Comparaison pour les températures élevées
    tx_seuil_chaud = 25
    heures_Observations_chaud = np.sum(obs_hourly_annual > tx_seuil_chaud)
    heures_modele_chaud = np.sum(mod_hourly_annual > tx_seuil_chaud)
    
    if heures_Observations_chaud > heures_modele_chaud:
        phrase_tx_chaud = f"Les observations ont plus d'heures avec une T>{tx_seuil_chaud}°C ({heures_Observations_chaud}) que le modèle ({heures_modele_chaud})."
    else:
        phrase_tx_chaud = f"Le modèle a plus d'heures avec une T>{tx_seuil_chaud}°C ({heures_modele_chaud}) que les observations ({heures_Observations_chaud})."

    tn_seuil_froid = 5
    heures_Observations_froid = np.sum(obs_hourly_annual < tn_seuil_froid)
    heures_modele_froid = np.sum(mod_hourly_annual < tn_seuil_froid)
    
    if heures_Observations_froid > heures_modele_chaud:
        phrase_tn_froid = f"Le modèle a plus d'heures avec une T<{tn_seuil_froid}°C ({heures_modele_froid}) que les observations ({heures_Observations_froid})."
    else:
        phrase_tn_froid = f"Les observations ont plus d'heures avec une T<{tn_seuil_froid}°C ({heures_Observations_froid}) que le modèle ({heures_modele_froid})."

    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_hist"] = [phrase_tx_chaud, phrase_tn_froid]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif histogrammes horaires/annuels")
    for p in st.session_state["resume_hist"]:
        st.write("- " + p)


    # ============================
    #   COURBES Tn / Tmoy / Tx
    # ============================
    st.subheader("Évolution mensuelle : Tn_mois / Tmoy_mois / Tx_mois (Modèle vs Observations)")
    st.markdown(
        """
        - Tn_mois et Tx_mois sont respectivement la températures minimale et maximal du mois
        - Les valeurs tracées représentent les températures minimales et maximales **absolues** du mois (c’est-à-dire P0 et P100)
        - De ce fait, les températures du mois ne dépassent jamais les bornes définies par Tn_mois et Tx_mois.
        - La température moyenne (Tmoy_mois) correspond à la moyenne mensuelle calculée sur l’ensemble des pas de temps. 
        - **Tmoy est différente de Tm = (Tn+Tx)/2**
        """,
        unsafe_allow_html=True
    )
    # Calcul des Tn/Tmoy/Tx pour 12 mois
    results_tstats = []
    for mois_num in range(1, 12+1):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
        obs_tn = np.min(obs_vals)
        obs_tm = np.mean(obs_vals)
        obs_tx = np.max(obs_vals)
    
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]
        mod_tn = np.min(mod_vals)
        mod_tm = np.mean(mod_vals)
        mod_tx = np.max(mod_vals)
    
        results_tstats.append({
            "Mois": mois,
            "Observations_Tn": obs_tn, "Modèle_Tn": mod_tn, "Observations_Tm": obs_tm, "Modèle_Tm": mod_tm, "Observations_Tx": obs_tx, "Modèle_Tx": mod_tx
        })
    
    df_tstats = pd.DataFrame(results_tstats)
    
    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14,4))

    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tx"], color="red", label="Modèle Tx", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tm"], color="white", label="Modèle Tmoy", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tn"], color="cyan", label="Modèle Tn", linestyle="-")

    ax.plot(df_tstats["Mois"], df_tstats["Observations_Tx"], color="red", label="Observations Tx", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats["Observations_Tm"], color="white", label="Observations Tmoy", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats["Observations_Tn"], color="cyan", label="Observations Tn", linestyle="--")

    ax.set_title(f"Tn_mois / Tmoy_mois / Tx_mois – Modèle vs Observations {file_sel}")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(facecolor="black")

    fig_tn_tx_mois = fig
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Tableau correspondant ----
    st.write("Tableau Tn_mois / Tmoy_mois / Tx_mois")
    st.dataframe(df_tstats.round(2), hide_index=True)

    # ---- Tableau des différences (Modèle - Observations) ----
    df_diff = pd.DataFrame({
        "Mois": df_tstats["Mois"],
        "Diff_Tn_mois": df_tstats["Modèle_Tn"] - df_tstats["Observations_Tn"],
        "Diff_Tmoy_mois": df_tstats["Modèle_Tm"] - df_tstats["Observations_Tm"],
        "Diff_Tx_mois": df_tstats["Modèle_Tx"] - df_tstats["Observations_Tx"],
    })
    
    df_diff_round = df_diff.copy()
    df_diff_round[["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]] = df_diff_round[["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]].round(2)
    
    st.write("Différences Modèle - Observations (Tn_mois / Tmoy_mois / Tx_mois)")
        
    # ---- Coloration avec background_gradient ----
    st.dataframe(
        df_diff_round.style
            .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT)
            .format("{:.2f}", subset=["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]),
        hide_index=True,
        use_container_width=True
    )

    # =============================
    # Comparaison moyenne annuelle
    # =============================
    
    # Moyenne annuelle sur 12 mois pour Observations et Modèle
    mean_Observations_Tx = df_tstats["Observations_Tx"].mean()
    mean_Model_Tx = df_tstats["Modèle_Tx"].mean()
    
    mean_Observations_Tm = df_tstats["Observations_Tm"].mean()
    mean_Model_Tm = df_tstats["Modèle_Tm"].mean()
    
    mean_Observations_Tn = df_tstats["Observations_Tn"].mean()
    mean_Model_Tn = df_tstats["Modèle_Tn"].mean()
    
    # Générer les phrases
    if mean_Observations_Tx > mean_Model_Tx:
        phrase_Tx = "En moyenne, les observations sont plus chaudes que le modèle pour les températures maximales (Tx_mois)."
    else:
        phrase_Tx = "En moyenne, le modèle est plus chaud que les observations pour les températures maximales (Tx_mois)."
    
    if mean_Observations_Tm > mean_Model_Tm:
        phrase_Tm = "En moyenne, les observations sont plus chaudes que le modèle pour les températures moyennes (Tmoy_mois)."
    else:
        phrase_Tm = "En moyenne, le modèle est plus chaud que les observations pour les températures moyennes (Tmoy_mois)."
    
    if mean_Observations_Tn > mean_Model_Tn:
        phrase_Tn = "En moyenne, les observations sont plus chaudes que le modèle pour les températures minimales (Tn_mois)."
    else:
        phrase_Tn = "En moyenne, le modèle est plus chaud que les observations pour les températures minimales (Tn_mois)."
    
    # Stocker dans st.session_state pour pouvoir les réutiliser dans la page Résumé
    st.session_state["resume_temp"] = [phrase_Tx, phrase_Tm, phrase_Tn]
    
    # Optionnel : afficher directement les phrases sur cette page
    st.subheader("Résumé comparatif annuel des températures")
    for p in st.session_state["resume_temp"]:
        st.write("- " + p)


    # ============================
    #  SECTION: Tn / Tmoy / Tx journaliers
    # ============================
    st.subheader("Tn_jour / Tmoy_jour / Tx_jour — CDF par mois et tableaux de percentiles")

    st.markdown(
        """
        - Tn_jour et Tx_jour sont respectivement la température minimale et maximale de la journée
        - Ici, les températures Tn_jour Tmoy_jour et Tx_jour sont décrites selon leurs percentiles pour chaque mois. Chaque mois compte 30 Tn,Tmoy et Tx.
        - **Exemple Tn_jour** : La valeur du P25 de Tn_jour du mois signifie que 25% des températures minimales journalières du mois sont inférieures à cette valeur.
        - **Exemple Tx_jour** : La valeur du P75 de Tx_jour du mois signifie que 75% des températures maximales journalières du mois sont inférieures à cette valeur.
        - Par conséquent, P99 des Tn_jour corresond au maximal des minimales journalières du mois
        
        """,
        unsafe_allow_html=True
    )
    
    def daily_stats_from_hourly(hourly):
        """
        Retourne trois tableaux journaliers (min, mean, max).
        Tronque si nécessaire pour avoir des jours complets (24h).
        """
        if len(hourly) < 24:
            return np.array([]), np.array([]), np.array([])
        n_full_days = len(hourly) // 24
        arr = np.array(hourly[: n_full_days * 24]).reshape((n_full_days, 24))
        daily_min = arr.min(axis=1)
        daily_mean = arr.mean(axis=1)
        daily_max = arr.max(axis=1)
        return daily_min, daily_mean, daily_max

    def count_days_in_bins(daily_values, bin_edges):
        """
        Retourne un tableau : nombre de jours dont la valeur tombe dans [X, X+1[ pour chaque X.
        """
        return np.histogram(daily_values, bins=bin_edges)[0]

    
    # percentiles pour les petits tableaux
    pct_table = percentiles_list  
    pct_for_cdf = np.linspace(0, 100, 100) 
    
    Tx_jour_all = []
    Tn_jour_all = []
    Tm_jour_all = []

    Tx_jour_mod_all = []
    Tn_jour_mod_all = []
    Tm_jour_mod_all = []
    
    # boucle mois par mois
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # ---- extraire hourly pour le mois: Observations (obs) et modèle (csv) ----
        obs_hourly = obs_mois_all[mois_num - 1] if len(obs_mois_all) >= mois_num else np.array([])
        idx0 = sum(heures_par_mois[:mois_num - 1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = model_values[idx0:idx1]
    
        # ---- calculer stats journalières ----
        obs_tn, obs_tm, obs_tx = daily_stats_from_hourly(obs_hourly)
        mod_tn, mod_tm, mod_tx = daily_stats_from_hourly(model_hourly)
        
        # Stocker les séries journalières OBS uniquement
        Tn_jour_all.append(obs_tn)
        Tm_jour_all.append(obs_tm)
        Tx_jour_all.append(obs_tx)

        # Stocker les séries journalières Modèle
        Tn_jour_mod_all.append(mod_tn)
        Tm_jour_mod_all.append(mod_tm)
        Tx_jour_mod_all.append(mod_tx)
    
        # Si pas de données, passer
        if obs_tn.size == 0 or mod_tn.size == 0:
            st.write(f"{mois} — données insuffisantes pour calculer les statistiques journalières.")
            continue
    
        # ---- préparer CDFs (percentiles des séries journalières) ----
        obs_tn_cdf = np.percentile(obs_tn, pct_for_cdf)
        mod_tn_cdf = np.percentile(mod_tn, pct_for_cdf)
        obs_tm_cdf = np.percentile(obs_tm, pct_for_cdf)
        mod_tm_cdf = np.percentile(mod_tm, pct_for_cdf)
        obs_tx_cdf = np.percentile(obs_tx, pct_for_cdf)
        mod_tx_cdf = np.percentile(mod_tx, pct_for_cdf)
    
        # ---- tracé : un seul graphique regroupant Tn / Tmoy / Tx ----
        fig, ax = plt.subplots(figsize=(12, 4))
    
        # Couleurs cohérentes pour chaque variable
        colors = {
            "Tn": "cyan",
            "Tm": "white",
            "Tx": "red"
        }
    
        # Tracer Modèle
        ax.plot(pct_for_cdf, mod_tx_cdf, linestyle="-", linewidth=2, label="Modèle Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, mod_tm_cdf, linestyle="-", linewidth=2, label="Modèle Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, mod_tn_cdf, linestyle="-", linewidth=2, label="Modèle Tn", color=colors["Tn"])
    
        # Tracer Observations
        ax.plot(pct_for_cdf, obs_tx_cdf, linestyle="--", linewidth=1.7, label="Observations Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, obs_tm_cdf, linestyle="--", linewidth=1.7, label="Observations Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, obs_tn_cdf, linestyle="--", linewidth=1.7, label="Observations Tn", color=colors["Tn"])
    
        # Mise en forme
        ax.set_title(f"{mois} — CDF Tn_jour / Tmoy_jour / Tx_jour (Modèle vs Observations {file_sel})", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
    
        st.pyplot(fig)
        plt.close(fig)
    
        def pct_table_values(arr, pct_list):
            return [np.percentile(arr, p) for p in pct_list]
    
        # ---- Tableau des percentiles ----
        tab = pd.DataFrame({
            "Percentile": [f"P{p}" for p in pct_table],
            "Observations_Tn": np.round(pct_table_values(obs_tn, pct_table), 2),
            "Mod_Tn": np.round(pct_table_values(mod_tn, pct_table), 2),
            "Observations_Tm": np.round(pct_table_values(obs_tm, pct_table), 2),
            "Mod_Tm": np.round(pct_table_values(mod_tm, pct_table), 2),
            "Observations_Tx": np.round(pct_table_values(obs_tx, pct_table), 2),
            "Mod_Tx": np.round(pct_table_values(mod_tx, pct_table), 2),
        })
    
        st.write(f"{mois} — Table des percentiles journaliers (Tn_jour / Tmoy_jour / Tx_jour)")
    
        num_cols = tab.select_dtypes(include=[np.number]).columns
        tab[num_cols] = tab[num_cols].apply(pd.to_numeric, errors="coerce")
        styler = tab.style.format({col: "{:.2f}" for col in num_cols})
        st.dataframe(styler, hide_index=True)
    
        # ---- Tableau des différences (Modèle - Observations) ----
        df_diff = pd.DataFrame({
            "Percentile": tab["Percentile"],
            "Diff_Tn_jour": tab["Mod_Tn"] - tab["Observations_Tn"],
            "Diff_Tm_jour": tab["Mod_Tm"] - tab["Observations_Tm"],
            "Diff_Tx_jour": tab["Mod_Tx"] - tab["Observations_Tx"],
        })
        
        # Redéfinir num_cols_diff avant l'utilisation
        num_cols_diff = ["Diff_Tn_jour", "Diff_Tm_jour", "Diff_Tx_jour"]
        
        # Convertir en float + arrondir
        df_diff[num_cols_diff] = df_diff[num_cols_diff].apply(pd.to_numeric, errors="coerce").round(2)

    
        st.write(f"{mois} — Différences Modèle - Observations (Tn_jour / Tmoy_jour / Tx_jour)")
    
        df_diff_styled = (
            df_diff.style
            .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT, subset=["Diff_Tn_jour","Diff_Tm_jour","Diff_Tx_jour"])
            .format({col: "{:.2f}" for col in ["Diff_Tn_jour","Diff_Tm_jour","Diff_Tx_jour"]})
        )
        st.dataframe(df_diff_styled, hide_index=True)

    # ---------------- Histogramme annuel Tn / Tx (Modèle vs Observations) ----------------
    st.subheader(f"Histogramme annuel Tn / Tx : Modèle et Observations {file_sel}")
    st.markdown(
        """
        La valeur de chaque barre correspond **au nombre de jours** dans lesquels la température
        minimale ou maximale journalière est comprise dans l’intervalle **[X°C , X+1°C[**.
        """,
        unsafe_allow_html=True
    )
    
    # --- Définition des classes de température ---
    bin_edges = np.arange(-10, 45, 1)  # Ajuste selon tes données
    bin_labels = bin_edges[:-1].astype(int)
    
    # --- Concaténation annuelle ---
    Tn_obs_annual = np.concatenate(Tn_jour_all)
    Tx_obs_annual = np.concatenate(Tx_jour_all)
    
    Tn_mod_annual = np.concatenate(Tn_jour_mod_all)
    Tx_mod_annual = np.concatenate(Tx_jour_mod_all)
    
    # --- Comptage dans les classes ---
    obs_counts_Tn = count_days_in_bins(Tn_obs_annual, bin_edges)
    mod_counts_Tn = count_days_in_bins(Tn_mod_annual, bin_edges)
    
    obs_counts_Tx = count_days_in_bins(Tx_obs_annual, bin_edges)
    mod_counts_Tx = count_days_in_bins(Tx_mod_annual, bin_edges)
    
    # --- Préparer DataFrame ---
    df_hist = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str) + "°C",
        "Obs_Tn": obs_counts_Tn,
        "Mod_Tn": mod_counts_Tn,
        "Obs_Tx": obs_counts_Tx,
        "Mod_Tx": mod_counts_Tx
    }).sort_values("Temp_Num")
    
    # ---------------- FIGURE Tn ----------------
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(df_hist["Temp_Num"] - 0.25, df_hist["Obs_Tn"], width=0.4,
           label="Observations Tn", color=couleur_Observations)
    ax.bar(df_hist["Temp_Num"] + 0.25, df_hist["Mod_Tn"], width=0.4,
           label="Modèle Tn", color=couleur_modele)
    
    ax.set_title("Histogramme annuel – Nombre de jours par classe de Tn")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre de jours")
    ax.legend(fontsize='large')
    st.pyplot(fig)
    plt.close(fig)

    pct_precision_Tn = precision_overlap(mod_counts_Tn, obs_counts_Tn)
    st.write(f"Précision du modèle sur les Tn_jour : **{pct_precision_Tn} %**")
    
    # ---------------- FIGURE Tx ----------------
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(df_hist["Temp_Num"] - 0.25, df_hist["Obs_Tx"], width=0.4,
           label="Observations Tx", color=couleur_Observations)
    ax.bar(df_hist["Temp_Num"] + 0.25, df_hist["Mod_Tx"], width=0.4,
           label="Modèle Tx", color=couleur_modele)
    
    ax.set_title("Histogramme annuel – Nombre de jours par classe de Tx")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre de jours")
    ax.legend(fontsize='large')
    st.pyplot(fig)
    plt.close(fig)

    pct_precision_Tx = precision_overlap(mod_counts_Tx, obs_counts_Tx)
    st.write(f"Précision du modèle sur les Tx_jour : **{pct_precision_Tx} %**")
     
    # --- Fonction nombre de jours de vague ---
    def nombre_jours_vague(T):
        T = np.array(T)
        n = len(T)
        jours_vague = np.zeros(n, dtype=bool)
        jours_vague[T >= 25.3] = True
        i = 0
        while i < n:
            if i + 2 < n and np.all(T[i:i+3] >= 23.4):
                debut = i
                fin = i + 2
                j = fin + 1
                while j < n and T[j] >= 23.4:
                    fin = j
                    j += 1
                prolong = fin + 1
                compteur = 0
                while prolong < n and compteur < 2:
                    if T[prolong] < 22.4:
                        break
                    fin = prolong
                    compteur += 1
                    prolong += 1
                jours_vague[debut:fin+1] = True
                i = fin + 1
            else:
                i += 1
        return int(jours_vague.sum()), jours_vague
    
    # ---------------- Calcul Tm et nombre de jours de vague sur l'année complète ----------------

    # 1) Longueur de chaque mois
    jours_par_mois = [len(Tx_jour_all[m]) for m in range(12)]
    
    # 2) Construire Tm sur toute l'année (continu)
    Tm_obs_all = np.concatenate([
        (np.array(Tx_jour_all[m]) + np.array(Tn_jour_all[m])) / 2 for m in range(12)
    ])
    
    Tm_mod_all = np.concatenate([
        (np.array(Tx_jour_mod_all[m]) + np.array(Tn_jour_mod_all[m])) / 2 for m in range(12)
    ])
    
    # 3) Calculer les jours de vague en continu sur l'année
    _, jours_vague_obs_all = nombre_jours_vague(Tm_obs_all)
    _, jours_vague_mod_all = nombre_jours_vague(Tm_mod_all)
    
    # 4) Re-découper par mois
    jours_vague_obs = []
    jours_vague_mod = []
    
    idx = 0
    for L in jours_par_mois:
        jours_vague_obs.append(int(jours_vague_obs_all[idx:idx+L].sum()))
        jours_vague_mod.append(int(jours_vague_mod_all[idx:idx+L].sum()))
        idx += L
    
    # ---------------- Tableau ----------------
    df_vagues = pd.DataFrame({
        "Mois": df_tstats["Mois"],
        "Observations": jours_vague_obs,
        "Modèle": jours_vague_mod
    })
    st.subheader("Nombre de jours de vague de chaleur par mois")
    st.dataframe(df_vagues, hide_index=True, use_container_width=True)
    
    # ---------------- Graphique bâtons ----------------
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 13)
    ax.bar(x - 0.2, jours_vague_obs, width=0.4, label="Observations", color=couleur_Observations)
    ax.bar(x + 0.2, jours_vague_mod, width=0.4, label="Modèle", color=couleur_modele)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de jours de vague de chaleur")
    ax.set_title("Nombre de jours de vague de chaleur par mois : Observations vs Modèle")
    ax.set_xticks(x)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    
    # ============================
    # GRAPHIQUES : Jours chauds et nuits tropicales par mois
    # ============================

    st.subheader("Graphiques : jours chauds et nuits tropicales par mois")

    # Choix seuil pour Tx
    tx_seuil = st.number_input("Seuil Tx_jour (°C) pour jours chauds :", value=25, step=1)
    tn_seuil = st.number_input("Seuil Tn_jour (°C) (Par défaut 20°C correspondant au seuil d'une nuit tropicale) :", value=20, step=1) 
    
    # Préparer listes pour stocker les valeurs par mois
    jours_chauds_Observations = []
    jours_chauds_modele = []
    nuits_tropicales_Observations = []
    nuits_tropicales_modele = []
    
    jours_chauds_total_Observations = 0
    jours_chauds_total_modele = 0
    nuits_tropicales_total_Observations = 0
    nuits_tropicales_total_modele = 0
    
    for mois_num in range(1, 13):
        # Observations
        obs_tx_jour = Tx_jour_all[mois_num - 1]
        obs_tn_jour = Tn_jour_all[mois_num - 1]
        jours_tx = np.sum(obs_tx_jour > tx_seuil)
        nuits_trop = np.sum(obs_tn_jour > tn_seuil)
        jours_chauds_Observations.append(jours_tx)
        nuits_tropicales_Observations.append(nuits_trop)
        jours_chauds_total_Observations += jours_tx
        nuits_tropicales_total_Observations += nuits_trop
    
        # Modèle
        mod_tx_jour = Tx_jour_mod_all[mois_num - 1]
        mod_tn_jour = Tn_jour_mod_all[mois_num - 1]
        jours_tx_mod = np.sum(mod_tx_jour > tx_seuil)
        nuits_trop_mod = np.sum(mod_tn_jour > tn_seuil)
        jours_chauds_modele.append(jours_tx_mod)
        nuits_tropicales_modele.append(nuits_trop_mod)
        jours_chauds_total_modele += jours_tx_mod
        nuits_tropicales_total_modele += nuits_trop_mod


    # Labels pour les mois
    mois_labels = [mois_noms[m] for m in range(1, 13)]
    x = np.arange(len(mois_labels))

    # ===== TABLEAU JOURS CHAUDS =====
    df_jours_chauds = pd.DataFrame({
        "Mois": mois_labels,
        "Observations": jours_chauds_Observations,
        "Modèle": jours_chauds_modele,
    })
    df_jours_chauds["Différence (Modèle - Obs)"] = df_jours_chauds["Modèle"] - df_jours_chauds["Observations"]
    

    st.markdown("Jours chauds par mois")
    st.dataframe(
        df_jours_chauds.style.background_gradient(
            cmap="bwr",
            subset=["Différence (Modèle - Obs)"],
            vmin=vminJ,
            vmax=vmaxJ
        ),
        hide_index=True,
        use_container_width=True
    )
    
    
    # ===== TABLEAU NUITS TROP =====
    df_nuits_trop = pd.DataFrame({
        "Mois": mois_labels,
        "Observations": nuits_tropicales_Observations,
        "Modèle": nuits_tropicales_modele,
    })
    df_nuits_trop["Différence (Modèle - Obs)"] = df_nuits_trop["Modèle"] - df_nuits_trop["Observations"]
    
    st.markdown("Nuits tropicales par mois")
    st.dataframe(
        df_nuits_trop.style.background_gradient(
            cmap="bwr",
            subset=["Différence (Modèle - Obs)"],
            vmin=vminJ,
            vmax=vmaxJ
        ),
        hide_index=True,
        use_container_width=True
    )

    # ---- Diagramme jours chauds ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.25, jours_chauds_Observations, width=0.4, color=couleur_Observations, label="Observations")
    ax.bar(x + 0.25, jours_chauds_modele, width=0.4, color=couleur_modele, label="Modèle")
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de jours Tx_jour > {tx_seuil}°C")
    ax.set_title("Jours chauds par mois")
    ax.legend(fontsize = "large")
    fig_jourschaud=fig
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Diagramme nuits tropicales ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.25, nuits_tropicales_Observations, width=0.4, color=couleur_Observations, label="Observations")
    ax.bar(x + 0.25, nuits_tropicales_modele, width=0.4, color=couleur_modele, label="Modèle")
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de nuits Tn_jour > {tn_seuil}°C")
    ax.set_title("Nuits tropicales par mois")
    ax.legend(fontsize = "large")
    fig_nuittrop=fig
    st.pyplot(fig)
    plt.close(fig)
    
    # =============================
    # Comparaison annuelle jours chauds / nuits tropicales
    # =============================
    
    # Jours chauds
    if jours_chauds_total_Observations > jours_chauds_total_modele:
        phrase_jours = f"les observations enregistrent plus de jours avec Tx>{tx_seuil}°C sur l'année ({jours_chauds_total_Observations}) que le modèle ({jours_chauds_total_modele})."
    else:
        phrase_jours = f"Le modèle enregistre plus de jours chauds avec Tx>{tx_seuil}°C sur l'année ({jours_chauds_total_modele}) que Observations ({jours_chauds_total_Observations})."
    
    # Nuits tropicales
    if nuits_tropicales_total_Observations > nuits_tropicales_total_modele:
        phrase_nuits = f"Les observations enregistrent plus de jours avec Tn>{tn_seuil}°C sur l'année ({nuits_tropicales_total_Observations}) que le modèle ({nuits_tropicales_total_modele})."
    else:
        phrase_nuits = f"Le modèle enregistre plus de jours avec Tn>{tn_seuil}°C sur l'année ({nuits_tropicales_total_modele}) que Observations ({nuits_tropicales_total_Observations})."
    
    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_chaud_nuit"] = [phrase_jours, phrase_nuits]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif jours chauds / nuits tropicales")
    for p in st.session_state["resume_chaud_nuit"]:
        st.write("- " + p)
   
    # ============================
    # Calcul DJC (chauffage) et DJF (froid)
    # ============================
    
    st.subheader("DJC (chauffage) et DJF (froid) journaliers — Observations vs Modèle")
    
    T_base_chauffage = float(st.text_input("Base DJC (°C) — chauffage", "19"))
    T_base_froid = float(st.text_input("Base DJF (°C) — refroidissement", "23"))
    
    results_djc = []
    results_djf = []
    mois_noms_sans_num = {
    1: "Janvier",   2: "Février",  3: "Mars",
    4: "Avril",     5: "Mai",      6: "Juin",
    7: "Juillet",   8: "Août",     9: "Septembre",
    10: "Octobre", 11: "Novembre", 12: "Décembre"
    }

    for mois_num in range(1, 13):
        mois = mois_noms_sans_num[mois_num]
    
        # Séries journalières déjà calculées
        Tx_Observations = Tx_jour_all[mois_num-1]
        Tn_Observations = Tn_jour_all[mois_num-1]
    
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = model_values[idx0:idx1]
        Tx_mod, Tm_mod, Tn_mod = daily_stats_from_hourly(model_hourly)
    
        DJC_Observations_jours, DJF_Observations_jours = [], []
        DJC_mod_jours, DJF_mod_jours = [], []
    
        n_jours = len(Tx_Observations)
        for j in range(n_jours):
            Tm_Observations = (Tx_Observations[j] + Tn_Observations[j]) / 2
            DJC_Observations_jours.append(max(0, T_base_chauffage - Tm_Observations))
            DJF_Observations_jours.append(max(0, Tm_Observations - T_base_froid))
    
            if j < len(Tx_mod):
                Tm_mod = (Tx_mod[j] + Tn_mod[j]) / 2
                DJC_mod_jours.append(max(0, T_base_chauffage - Tm_mod))
                DJF_mod_jours.append(max(0, Tm_mod - T_base_froid))
            else:
                DJC_mod_jours.append(0)
                DJF_mod_jours.append(0)
    
        DJC_Observations_sum = float(np.nansum(DJC_Observations_jours))
        DJC_mod_sum = float(np.nansum(DJC_mod_jours))
        DJF_Observations_sum = float(np.nansum(DJF_Observations_jours))
        DJF_mod_sum = float(np.nansum(DJF_mod_jours))
    
        results_djc.append({
            "Mois": mois,
            "Observations": DJC_Observations_sum,
            "Modèle": DJC_mod_sum,
            "Différence": DJC_mod_sum - DJC_Observations_sum
        })
        results_djf.append({
            "Mois": mois,
            "Observations": DJF_Observations_sum,
            "Modèle": DJF_mod_sum,
            "Différence": DJF_mod_sum - DJF_Observations_sum
        })
    
    df_DJC = pd.DataFrame(results_djc).fillna(0)
    df_DJF = pd.DataFrame(results_djf).fillna(0)
    
    # Convertir explicitement les colonnes numériques en float
    for df in [df_DJC, df_DJF]:
        for col in ["Observations", "Modèle", "Différence"]:
            df[col] = df[col].astype(float)
    
    # --------------------------
    # Affichage tables Streamlit
    # --------------------------
    st.subheader("DJC – Chauffage (somme journalière par mois)")
    st.dataframe(df_DJC.round(2))  # Arrondi à 2 décimales
    
    st.subheader("DJF – Refroidissement (somme journalière par mois)")
    st.dataframe(df_DJF.round(2))  # Arrondi à 2 décimales

    
    # --------------------------
    # Diagrammes bâtons mensuels
    # --------------------------
    st.subheader("Diagrammes bâtons mensuels — DJC et DJF")

    # Convertir en DataFrames
    df_DJC = pd.DataFrame(results_djc)
    df_DJF = pd.DataFrame(results_djf)
    
    # -----------------------------
    # Diagrammes en bâtons par mois
    # -----------------------------
    figures = {}   # dictionnaire où on stocke les figures

    for df, titre in zip([df_DJC, df_DJF], ["DJC", "DJF"]):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df.index - 0.25, df["Observations"], width=0.4,
               color=couleur_Observations, label="Observations")
        ax.bar(df.index + 0.25, df["Modèle"], width=0.4,
               color=couleur_modele, label="Modèle")
    
        ax.set_xticks(df.index)
        ax.set_xticklabels(df["Mois"])
        ax.set_title(f"{titre} mensuel — Modèle vs Observations")
        ax.set_ylabel(f"{titre} (°C·jour)")
        ax.set_xlabel("Mois")
        ax.legend(fontsize = "large")
    
        # enregistrer la figure dans le dictionnaire
        figures[titre] = fig
    
        st.pyplot(fig)
        plt.close(fig)

    # --------------------------
    # Somme annuelle DJC et DJF
    # --------------------------
    total_DJC_Observations = df_DJC["Observations"].sum()
    total_DJC_modele = df_DJC["Modèle"].sum()
    
    total_DJF_Observations = df_DJF["Observations"].sum()
    total_DJF_modele = df_DJF["Modèle"].sum()
    
    st.subheader("Sommes annuelles")
    st.write(f"DJC annuel : Observations = {total_DJC_Observations:.0f}    /    Modèle = {total_DJC_modele:.0f}")
    st.write(f"DJF annuel : Observations = {total_DJF_Observations:.0f}    /    Modèle = {total_DJF_modele:.0f}")

    # =============================
    # Résumé automatique DJC / DJF
    # =============================
    
    # DJC (chauffage)
    if total_DJC_Observations > total_DJC_modele:
        phrase_djc = f"Les observations ont une demande de chauffage annuelle plus élevée ({total_DJC_Observations:.0f} °C·jour) que le modèle ({total_DJC_modele:.0f} °C·jour)."
    elif total_DJC_modele > total_DJC_Observations:
        phrase_djc = f"Le modèle a une demande de chauffage annuelle plus élevée ({total_DJC_modele:.0f} °C·jour) que Observations ({total_DJC_Observations:.0f} °C·jour)."
    else:
        phrase_djc = "Observations et le modèle ont la même demande de chauffage annuelle."
    
    # DJF (refroidissement)
    if total_DJF_Observations > total_DJF_modele:
        phrase_djf = f"Les observations ont une demande de refroidissement annuelle plus élevée ({total_DJF_Observations:.0f} °C·jour) que le modèle ({total_DJF_modele:.0f} °C·jour)."
    elif total_DJF_modele > total_DJF_Observations:
        phrase_djf = f"Le modèle a une demande de refroidissement annuelle plus élevée ({total_DJF_modele:.0f} °C·jour) que Observations ({total_DJF_Observations:.0f} °C·jour)."
    else:
        phrase_djf = "Observations et le modèle ont la même demande de refroidissement annuelle."
    
    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_djc_djf"] = [phrase_djc, phrase_djf]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif DJC / DJF")
    for p in st.session_state["resume_djc_djf"]:
        st.write("- " + p)
 
    # ======================================
    #  COURBES DES PERCENTILES PAR MOIS
    # ======================================
    st.subheader("Évolution mensuelle des percentiles (Modèle vs Observations)")

    df_percentiles_all = []
    percentiles_list2 = [10,50,90]
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
    
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]

        
        # Ajout des percentiles
        for p in percentiles_list2:
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": np.percentile(obs_vals, p),
                "Mod": np.percentile(mod_vals, p)
            })

    # Table ordonnée pour faciliter les tracés
    df_percentiles_ordered = (
        pd.DataFrame(df_percentiles_all)
        .assign(Pnum=lambda d: d["Percentile"].str.extract("(\d+)").astype(int))
        .sort_values(["Pnum", "Mois"])
    )
    
    # Construction du graphique par percentile
    fig, ax = plt.subplots(figsize=(14,5))
    colors_perc = ["darkcyan", "khaki", "firebrick"]
    i=0
    for p in percentiles_list2:
        dfp = df_percentiles_ordered[df_percentiles_ordered["Pnum"] == p]
        # Observations : ligne pointillée
        ax.plot(
            dfp["Mois"], dfp["Obs"],
            linestyle="--", label=f"Observations P{p}", color=colors_perc[i]
        )
        # Modèle : ligne pleinne
        ax.plot(
            dfp["Mois"], dfp["Mod"],
            linestyle="-", label=f"Modèle P{p}", color=colors_perc[i]
        )
        i+=1
    
    ax.set_title(f"Percentiles {percentiles_list2} – Modèle vs Observations {file_sel}")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2, facecolor="black")
    st.pyplot(fig)
    plt.close(fig)

    # -------- Calcul des percentiles P1 à P99 --------
    percentiles = np.arange(1, 100)
    P_obs = np.percentile(obs_annee, percentiles)
    P_mod = np.percentile(model_annee, percentiles)
    
    # -------- Graphique PXX modèle vs TRACC avec croix et couleurs conditionnelles --------
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Définir les couleurs selon qui est plus chaud
    colors = [couleur_Observations if obs > mod else couleur_modele for obs, mod in zip(P_obs, P_mod)]
    
    # Tracer les croix
    ax.scatter(P_obs, P_mod, color=colors, marker='x', s=25, label='Percentiles')
    
    # Diagonale y=x
    min_val = min(min(P_obs), min(P_mod))
    max_val = max(max(P_obs), max(P_mod))
    ax.plot([min_val, max_val], [min_val, max_val], color='white', linestyle='--', label='y=x')
    
    # Carré : même échelle sur x et y
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', 'box')
    
    ax.set_xlabel("Températures des observations (°C)")
    ax.set_ylabel("Températures du modèle (°C)")
    ax.set_title("Comparaison des percentiles annuels")
    ax.grid(True, linestyle=':', color='gray', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
        - **Analyse** : Si les croix sont sous la diagonale, alors le modèle est plus froid (et inversement)
        
        Ce diagramme quantiles-quantiles représente, pour chaque percentile, les valeurs de température issues des observations et celles issues du modèle.
        Ce type de représentation permet de comparer directement les deux sources de données sur l’ensemble de la distribution.
        Chaque croix represente les températures du même percentile (de P1 à P99).
        Les coordonnées de chaque percentile sont définies par (x;y) = (T°C observations ; T°C modèle)
        Ce diagramme permet alors de comparer les valeurs de températures des deux sources de donnée pour le même percentile.
        **Interprétation** : L’écart par rapport à la diagonale permet ainsi de quantifier le biais du modèle sur l’ensemble de la distribution, 
        en mettant en évidence d’éventuelles dérives spécifiques aux basses, moyennes ou hautes valeurs de température.
        """,
        unsafe_allow_html=True
    )


    # -------- Graphiques CDF et percentiles --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")
    df_percentiles_all = []
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_mois = obs_mois_all[mois_num-1]
        mod_mois = model_values[sum(heures_par_mois[:mois_num-1]):sum(heures_par_mois[:mois_num])]
        obs_percentiles_100 = np.percentile(obs_mois, np.linspace(0, 100, 100))
        mod_percentiles_100 = np.percentile(mod_mois, np.linspace(0, 100, 100))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.linspace(0, 100, 100), mod_percentiles_100, label="Modèle", color=couleur_modele)
        ax.plot(np.linspace(0, 100, 100), obs_percentiles_100, label=f"Observations {file_sel}", color=couleur_Observations)
        ax.set_title(f"{mois} - Fonction de répartition", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
        st.pyplot(fig)
        plt.close(fig)

        obs_p = np.percentile(obs_mois, percentiles_list)
        mod_p = np.percentile(mod_mois, percentiles_list)
        df_p = pd.DataFrame({
            "Percentile": [f"P{p}" for p in percentiles_list],
            f"Observations {file_sel}": obs_p,
            "Modèle": mod_p
        }).round(2)
        st.write(f"{mois} - Percentiles")
        st.dataframe(df_p, hide_index=True)

        for i, p in enumerate(percentiles_list):
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": obs_p[i],
                "Mod": mod_p[i]
            })

    # -------- Fonction de répartition ANNUELLE --------
    st.subheader("Fonction de répartition annuelle (CDF)")
    
    # Regroupement annuel
    obs_annual = np.concatenate(obs_mois_all)         # Observations Observations - toutes les heures de l'année
    mod_annual = model_values                         # Modèle : déjà toutes les heures
    
    # Percentiles pour CDF (0–100)
    percentiles_cdf = np.linspace(0, 100, 100)
    obs_percentiles_annual = np.percentile(obs_annual, percentiles_cdf)
    mod_percentiles_annual = np.percentile(mod_annual, percentiles_cdf)
    
    # ----- Plot de la CDF annuelle -----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(percentiles_cdf, mod_percentiles_annual, label="Modèle", color=couleur_modele)
    ax.plot(percentiles_cdf, obs_percentiles_annual, label=f"Observations {file_sel}", color=couleur_Observations)
    
    ax.set_title("Année entière - Fonction de répartition", color="white")
    ax.set_xlabel("Percentile", color="white")
    ax.set_ylabel("Température (°C)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black")
    ax.set_facecolor("none")
    
    fig_cdf = fig
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ------ Tableau des percentiles annuels ------
    obs_p_annual = np.percentile(obs_annual, percentiles_list)
    mod_p_annual = np.percentile(mod_annual, percentiles_list)
    
    df_p_annual = pd.DataFrame({
        "Percentile": [f"P{p}" for p in percentiles_list],
        f"Observations {file_sel}": obs_p_annual,
        "Modèle": mod_p_annual
    }).round(2)
    
    st.write("Année entière - Percentiles")
    st.dataframe(df_p_annual, hide_index=True)


    st.subheader(f"Bilan modèle vs Observations {file_sel} (Modèle - Observations)") 
    # Création du DataFrame
    df_bilan = pd.DataFrame(df_percentiles_all).round(2)
    df_bilan["Ecart"] = df_bilan["Mod"] - df_bilan["Obs"]
    # Extraire le numéro du percentile (5, 25, ...) pour imposer l'ordre
    df_bilan["Percentile_num"] = df_bilan["Percentile"].str.extract("(\d+)").astype(int)
    # Imposer l'ordre des percentiles
    df_bilan["Percentile"] = pd.Categorical(df_bilan["Percentile"], 
                                            categories=[f"P{p}" for p in percentiles_list], 
                                            ordered=True)
    # Pivot pour affichage
    df_bilan_pivot = df_bilan.pivot(index="Percentile", columns="Mois", values="Ecart").round(2)
    # Affichage stylé avec couleurs selon l'écart
    st.dataframe(
        df_bilan_pivot.style
        .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT)
        .format("{:.2f}")
    )
   
    # ---- Stockage des figures dans session_state ----
    st.session_state["fig_hist_year"] = fig_hist_year
    st.session_state["fig_hist_diff"] = fig_hist_diff
    st.session_state["df_rmse"] = df_rmse
    st.session_state["df_rmse_styled"] = df_rmse_styled
    st.session_state["fig_tn_tx_mois"] = fig_tn_tx_mois
    st.session_state["fig_jourschaud"] = fig_jourschaud
    st.session_state["fig_nuittrop"] = fig_nuittrop
    st.session_state["fig_cdf"] = fig_cdf
    st.session_state["fig_DJC"] = figures["DJC"]
    st.session_state["fig_DJF"] = figures["DJF"]

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:04:12 2025

@author: joris
"""

import streamlit as st

st.title("Résumé des résultats")

# Vérification
missing = [k for k in 
           ["fig_hist_year", "fig_hist_diff", "df_rmse", "df_rmse_styled",
            "fig_tn_tx_mois", "fig_jourschaud", "fig_nuittrop",
            "fig_cdf", "fig_DJC", "fig_DJF"]
           if k not in st.session_state]

if missing:
    st.error("Les données du résumé ne sont pas présentes. "
             "Retourne sur la page principale d'abord.")
    st.stop()

# --- Histogrammes annuels ---
st.subheader("Histogrammes annuels")
st.pyplot(st.session_state["fig_hist_year"])
st.pyplot(st.session_state["fig_hist_diff"])

st.subheader("Résumé histogrammes horaires / annuels")

if "resume_hist" in st.session_state:
    for p in st.session_state["resume_hist"]:
        st.write("- " + p)
else:
    st.write("Les données histogrammes ne sont pas encore calculées.")


# --- Tableau RMSE ---
st.subheader("Précision du modèle : RMSE et percentiles")
st.dataframe(st.session_state["df_rmse_styled"], hide_index=True)

# --- Figures mensuelles Tn/Tmoy/Tx ---
st.subheader("Évolution Tn/Tmoy/Tx")
st.pyplot(st.session_state["fig_tn_tx_mois"])
st.markdown("**Commentaires Tn/Tmoy/Tx**")
# Vérifier si les phrases existent dans st.session_state
if "resume_temp" in st.session_state:
    for p in st.session_state["resume_temp"]:
        st.write("- " + p)
else:
    st.write("Les données comparatives ne sont pas encore disponibles. Veuillez d'abord calculer les températures sur la page principale.")

st.subheader("Jours chauds")
st.pyplot(st.session_state["fig_jourschaud"])
st.subheader("Nuits tropicales")
st.pyplot(st.session_state["fig_nuittrop"])
st.markdown("**Commentaires jours chauds et nuits tropicales**")
st.subheader("Résumé jours chauds / nuits tropicales")

if "resume_chaud_nuit" in st.session_state:
    for p in st.session_state["resume_chaud_nuit"]:
        st.write("- " + p)
else:
    st.write("Les données sur les jours chauds et nuits tropicales ne sont pas encore calculées.")


# --- CDF annuelle ---
st.subheader("CDF annuelle")
st.pyplot(st.session_state["fig_cdf"])

# --- DJC et DJF ---
st.subheader("DJC")
st.pyplot(st.session_state["fig_DJC"])
st.subheader("DJF")
st.pyplot(st.session_state["fig_DJF"])

st.subheader("Résumé DJC / DJF")
st.markdown("**Commentaires DJC / DJF :**")


if "resume_djc_djf" in st.session_state:
    for p in st.session_state["resume_djc_djf"]:
        st.write("- " + p)
else:
    st.write("Les données DJC/DJF ne sont pas encore calculées.")




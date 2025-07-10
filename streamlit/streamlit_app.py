import mlflow
import streamlit as st
import os
import requests
import re
import json
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- Configuration
os.environ["HOME"] = "/tmp"
MLFLOW_TRACKING_URI = "https://sbiwi-mlflow-server-demo.hf.space"
EXPERIMENT_NAME = "Meteo"
ARTIFACT_FILENAME = "prediction_results.txt"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

st.title("🌍 Météo à Paris sur carte")

# --- Récupération de l'expérience
try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        st.error(f"Aucune expérience nommée '{EXPERIMENT_NAME}' n'a été trouvée.")
        st.stop()
    experiment_id = experiment.experiment_id
except Exception as e:
    st.error(f"Erreur récupération expérience : {e}")
    st.stop()

# --- Récupération du dernier run
runs = client.search_runs([experiment_id], order_by=["attribute.start_time DESC"], max_results=1)
if not runs:
    st.warning("Aucun run trouvé.")
    st.stop()

run = runs[0]
run_id = run.info.run_id

# --- Téléchargement du fichier
try:
    list_url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/artifacts/list"
    response = requests.get(list_url, params={"run_id": run_id}, timeout=10)
    files = response.json().get("files", [])
    if not any(f["path"] == ARTIFACT_FILENAME for f in files):
        st.warning(f"{ARTIFACT_FILENAME} introuvable.")
        st.stop()

    get_url = f"{MLFLOW_TRACKING_URI}/get-artifact"
    content = requests.get(get_url, params={"path": ARTIFACT_FILENAME, "run_uuid": run_id}, timeout=10).text

    # Extraction des infos utiles
    actual_match = re.search(r"Actual Weather:\s*(.+)", content)
    predicted_match = re.search(r"Predicted Weather:\s*(.+)", content)

    if actual_match and predicted_match:
        actual_weather = actual_match.group(1)
        predicted_weather = predicted_match.group(1)

        st.success("✅ Dernière prédiction météo")
        st.markdown(f"**Valeur réelle :** `{actual_weather}`")
        st.markdown(f"**Prédiction IA :** `{predicted_weather}`")

        # Choix de l’icône météo
        weather_icons = {
            #"Clear": "https://openweathermap.org/img/wn/01d.png",
            "Clear": "https://cdn-icons-png.flaticon.com/128/869/869869.png",
            "Clouds": "https://openweathermap.org/img/wn/03d.png",
            "Rain": "https://openweathermap.org/img/wn/09d.png",
            "Fog": "https://openweathermap.org/img/wn/50d.png",
            "Fod": "https://openweathermap.org/img/wn/50d.png",  # faute probable
        }
        icon_url = weather_icons.get(predicted_weather, "https://openweathermap.org/img/wn/01d.png")

        # Création de la carte centrée sur Paris
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=10, tiles="CartoDB Positron")

        folium.Marker(
            location=[48.8566, 2.3522],
            popup=f"Prévision : {predicted_weather}",
            icon=folium.CustomIcon(icon_url, icon_size=(50, 50))
        ).add_to(m)

        # Affichage de la carte dans Streamlit
        st.subheader("🗺️ Carte météo")
        st_folium(m, width=700, height=500)

    else:
        st.error("Impossible d'extraire la prédiction ou la valeur réelle.")

except Exception as e:
    st.error(f"Erreur API REST : {e}")

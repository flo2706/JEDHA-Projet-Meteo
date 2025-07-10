import streamlit as st
import pandas as pd
import boto3
import folium
from streamlit_folium import st_folium
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from babel.dates import format_datetime

# --- Configuration S3
BUCKET_NAME = "my-jedha-bucket"
OBJECT_KEY = "meteo_predict/weather_predictions.csv"

st.title("🌍 Météo en France — dernières prédictions")

# --- Téléchargement du CSV depuis S3
try:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=OBJECT_KEY)
    data = obj["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(data))
except Exception as e:
    st.error(f"Erreur de lecture depuis S3 : {e}")
    st.stop()

# --- Vérification du contenu
if df.empty:
    st.warning("Le fichier de prédictions est vide.")
    st.stop()

# --- Conversion des dates avec fuseau horaire français
df["execution_date"] = pd.to_datetime(df["execution_date"], utc=True)
df["execution_date"] = df["execution_date"].dt.tz_convert("Europe/Paris")

# --- Sélection de la dernière date
latest_date = df["execution_date"].max()
latest_df = df[df["execution_date"] == latest_date]

# --- Format de la date en français avec heure complète
formatted_date = format_datetime(latest_date, "d MMMM y 'à' HH:mm:ss", locale='fr_FR')

# --- Choix des icônes météo
weather_icons = {
    "Clear": "https://cdn-icons-png.flaticon.com/128/869/869869.png",
    "Clouds": "https://openweathermap.org/img/wn/03d.png",
    "Rain": "https://openweathermap.org/img/wn/09d.png",
    "Fog": "https://openweathermap.org/img/wn/50d.png",
    "Fod": "https://openweathermap.org/img/wn/50d.png",  # faute probable
}

# --- Carte centrée sur la France
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles="CartoDB Positron")

# --- Ajout des marqueurs météo
for _, row in latest_df.iterrows():
    icon_url = weather_icons.get(row["prediction"], weather_icons["Clear"])
    popup_html = f"""
        <b>{row['ville']}</b><br>
        Prédiction : {row['prediction']}<br>
        Réel : {row['valeur_reelle']}<br>
        Confiance : {row['confidence']*100:.1f}%
    """
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=popup_html,
        icon=folium.CustomIcon(icon_url, icon_size=(40, 40))
    ).add_to(m)

# --- Affichage de la carte avec la date locale formatée
st.subheader(f"🗺️ Prédictions météo du {formatted_date}")
st_folium(m, width=800, height=600)

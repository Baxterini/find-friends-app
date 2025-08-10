import streamlit as st
import pandas as pd
import plotly.express as px
# from pycaret.clustering import load_model, predict_model  # WYŁĄCZONE
import joblib  # zastąpienie dla PyCaret
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# === USTAWIENIA STRONY ===
st.set_page_config(page_title="Find Friends — V5+ (TEST)", page_icon="🤝", layout="wide")

# === ŚCIEŻKI I PLIKI ===
HERE = Path(__file__).parent
SEP = ';'
DATA_PATH = HERE / "welcome_survey_simple_v2.csv"
MODEL_NAME = "welcome_survey_clustering_pipeline_v2"
MODEL_PATH = HERE / f"{MODEL_NAME}.pkl"
CLUSTER_INFO_PATH = HERE / "welcome_survey_cluster_names_and_descriptions_v2.json"
COLS = ["age", "edu_level", "fav_animals", "fav_place", "gender"]

st.title("🚧 TEST BEZ PYCARET")
st.info("Testujemy czy aplikacja uruchomi się bez PyCaret. Model clustering jest tymczasowo wyłączony.")

# Podstawowe funkcje bez modelu
@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pd.DataFrame(columns=COLS).to_csv(DATA_PATH, index=False, sep=SEP)
    df = pd.read_csv(DATA_PATH, sep=SEP)
    return df

@st.cache_data
def load_cluster_info() -> dict:
    with open(CLUSTER_INFO_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)

# Proste UI
st.sidebar.header("Dane testowe")
age = st.sidebar.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
edu_level = st.sidebar.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])

# Test ładowania danych
try:
    df = load_data()
    st.success(f"✅ Dane załadowane: {len(df)} rekordów")
    if not df.empty:
        st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Błąd ładowania danych: {e}")

# Test ładowania JSON
try:
    cluster_info = load_cluster_info()
    st.success(f"✅ JSON załadowany: {len(cluster_info)} klastrów")
    st.json(cluster_info)
except Exception as e:
    st.error(f"❌ Błąd ładowania JSON: {e}")

st.info("Jeśli widzisz ✅ powyżej, to znaczy że problem był w PyCaret. Można przywrócić model z inną biblioteką.")
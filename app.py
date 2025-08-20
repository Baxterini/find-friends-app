import streamlit as st

st.set_page_config(page_title="Find Friends — V5+", page_icon="🤝", layout="wide")

import pandas as pd
import plotly.express as px
from pycaret.clustering import load_model, predict_model
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import sklearn, pycaret

st.caption(f"env → sklearn: {sklearn.__version__} | pycaret: {pycaret.__version__} | numpy: {np.__version__}")

# 🔎 Diagnostyka w sidebarze
st.sidebar.write("scikit-learn:", sklearn.__version__)
st.sidebar.write("pycaret:", pycaret.__version__)
st.sidebar.write("numpy:", np.__version__)

# === ŚCIEŻKI I PLIKI ===
HERE = Path(__file__).parent
SEP = ';'
DATA_PATH = HERE / "welcome_survey_simple_v2.csv"
MODEL_NAME = "welcome_survey_clustering_pipeline_v2"  # bez .pkl
MODEL_PATH = HERE / MODEL_NAME  # też bez .pkl — PyCaret doda sam
CLUSTER_INFO_PATH = HERE / "welcome_survey_cluster_names_and_descriptions_v2.json"
COLS = ["age", "edu_level", "fav_animals", "fav_place", "gender"]

# === FUNKCJE ===
@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))


# === KOLEJNOŚĆ KATEGORII (do mediany wieku) ===
AGE_ORDER = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']
AGE_RANK = {a: i for i, a in enumerate(AGE_ORDER)}

# === MINI HELPERY ===
def ensure_csv(path: Path):
    if not path.exists():
        pd.DataFrame(columns=COLS).to_csv(path, index=False, sep=SEP)

@st.cache_data
def load_data() -> pd.DataFrame:
    ensure_csv(DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep=SEP)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        st.error(f"Brakuje kolumn w CSV: {missing}. Spodziewane nagłówki: {COLS}")
        st.stop()
    return df

# MODELE = resource (nie data)
@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))

@st.cache_data
def load_cluster_info() -> dict:
    with open(CLUSTER_INFO_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def save_user_row(row: dict):
    df = load_data().copy()
    is_dup = (df[COLS] == pd.Series(row)).all(axis=1).any() if not df.empty else False
    if not is_dup:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(DATA_PATH, sep=SEP, index=False)
        load_data.clear()

def normalize_cluster_id(value) -> int:
    # Obsłuży 'Cluster 0' i 0
    return int(str(value).replace("Cluster", "").strip())

def cluster_name_desc(cluster_id, mapping: dict):
    block = mapping.get(str(cluster_id))
    if not block:
        return f"Klaster {cluster_id}", "Brak opisu dla tej grupy."
    if isinstance(block, dict):
        return block.get("name", f"Klaster {cluster_id}"), block.get("description", "Brak opisu dla tej grupy.")
    return str(block), "Brak opisu dla tej grupy."

def median_age_category(df: pd.DataFrame) -> str:
    """Mediana wieku po kolejności AGE_ORDER, odporna na dtype 'category'."""
    if df.empty or "age" not in df.columns:
        return "unknown"
    cat = pd.Categorical(df["age"], categories=AGE_ORDER, ordered=True)
    codes = pd.Series(cat.codes)
    codes = codes[codes >= 0]
    if codes.empty:
        return "unknown"
    med_code = int(np.median(codes.to_numpy()))
    med_code = max(0, min(med_code, len(AGE_ORDER)-1))
    return AGE_ORDER[med_code]

def percent_breakdown(series: pd.Series) -> pd.DataFrame:
    """Zwraca procentowy rozkład wartości (0-100, 1 miejsce po przecinku)."""
    if series.empty:
        return pd.DataFrame(columns=["kategoria", "procent"])
    pct = series.value_counts(normalize=True).sort_index() * 100.0
    out = pct.reset_index()
    out.columns = ["kategoria", "procent"]
    out["procent"] = out["procent"].round(1)
    return out

def plot_percent_bar(df_pct: pd.DataFrame, title: str, template: str):
    if df_pct.empty:
        st.info(f"Brak danych do wykresu: {title}")
        return
    fig = px.bar(df_pct, x="kategoria", y="procent", title=title, text="procent", template=template)
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

def radar_trait_scores(df_grp: pd.DataFrame) -> dict:
    """Pięć intuicyjnych wskaźników 0–1 dla grupy (do radaru)."""
    if df_grp.empty:
        return {"Młodsi":0,"Natura":0,"Zwierzęta":0,"Wyższe wyksz.":0,"Kobiety":0}
    score_mlodzi = df_grp['age'].isin(['<18','18-24','25-34']).mean()
    score_natura = df_grp['fav_place'].isin(['Nad wodą','W lesie','W górach']).mean()
    score_zwierz = df_grp['fav_animals'].isin(['Psy','Koty','Koty i Psy']).mean()
    score_wyzsze = (df_grp['edu_level'] == 'Wyższe').mean()
    score_kobiety = (df_grp['gender'] == 'Kobieta').mean()
    return {
        "Młodsi": float(score_mlodzi),
        "Natura": float(score_natura),
        "Zwierzęta": float(score_zwierz),
        "Wyższe wyksz.": float(score_wyzsze),
        "Kobiety": float(score_kobiety),
    }

def cluster_card(title: str, desc: str, cluster_id: int, count_same: int):
    st.markdown(f"""
<div class="cluster-card">
  <div class="big-title">🔮 Najbliżej Ci do grupy: {title}</div>
  <div style="margin:.25rem 0 0.5rem 0;">{desc}</div>
  <div class="subtle">ID klastra: <b>{cluster_id}</b> • Liczba osób w tej grupie: <b>{count_same}</b></div>
</div>
""", unsafe_allow_html=True)

# === SIDEBAR: motyw + formularz ===
if "theme" not in st.session_state:
    st.session_state.theme = "Jasny"

st.sidebar.header("Ustawienia")
st.session_state.theme = st.sidebar.selectbox("Motyw", ["Jasny", "Ciemny"], index=0)

# CSS dla motywu
if st.session_state.theme == "Ciemny":
    st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #f0f2f6; }
    .stMarkdown, .stText, .stCaption, .stDataFrame { color: #f0f2f6 !important; }
    .cluster-card {
      padding: 1rem 1.25rem; border-radius: 16px;
      background: linear-gradient(135deg, #1a2030 0%, #111827 100%);
      border: 1px solid rgba(255,255,255,.08);
      box-shadow: 0 8px 30px rgba(0,0,0,.35);
      color: #f0f2f6;
    }
    .big-title { font-size: 2rem; font-weight: 800; margin-bottom: .25rem; color: #ffffff; }
    .subtle { color: rgba(255,255,255,.65); }
    .footer { color: rgba(255,255,255,.55); font-size: .9rem; }
    </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly_dark"
else:
    st.markdown("""
    <style>
    .cluster-card {
      padding: 1rem 1.25rem; border-radius: 16px;
      background: linear-gradient(135deg, #f6f9ff 0%, #eef6ff 100%);
      border: 1px solid rgba(0,0,0,.06);
      box-shadow: 0 8px 30px rgba(0,0,0,.04);
    }
    .big-title { font-size: 2rem; font-weight: 800; margin-bottom: .25rem; }
    .subtle { color: rgba(0,0,0,.55); }
    .footer { color: rgba(0,0,0,.45); font-size: .9rem; }
    </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly"

st.sidebar.header("Powiedz nam coś o sobie")
st.sidebar.markdown("Dopasujemy Cię do osób o podobnych zainteresowaniach 🧭")

age = st.sidebar.selectbox("Wiek", AGE_ORDER)
edu_level = st.sidebar.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
fav_animals = st.sidebar.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
fav_place = st.sidebar.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
gender = st.sidebar.radio("Płeć", ['Mężczyzna', 'Kobieta'])

user_data = {
    'age': age,
    'edu_level': edu_level,
    'fav_animals': fav_animals,
    'fav_place': fav_place,
    'gender': gender
}

c1, c2 = st.sidebar.columns(2)
if c1.button("💾 Zapisz profil"):
    try:
        save_user_row(user_data)
        st.sidebar.success("Zapisano! 🙌")
    except Exception as e:
        st.sidebar.error(f"Błąd: {e}")

if c2.button("🔁 Wyczyść cache"):
    load_data.clear(); get_model.clear(); load_cluster_info.clear()
    st.sidebar.success("Cache wyczyszczony — uruchom ponownie obliczenia.")

# === HEADER ===
st.markdown("## 🫱🏻‍🫲🏻 Find Friends — V5+ (statystyki + motyw + radar)")
st.caption("Mediana wieku, rozkłady procentowe, radar oraz przełącznik motywu Jasny/Ciemny.")

# === LEKKA DIAGNOSTYKA ===
diag = []
def log(msg): diag.append(msg)

# === MODEL & DANE ===
try:
    log("Wczytuję model…")
    model = get_model()
    log("OK: model")
except Exception as e:
    st.error("Nie udało się wczytać modelu. Sprawdź artefakty / wersje pakietów.")
    st.exception(e); st.stop()

try:
    log("Wczytuję dane…")
    df = load_data()
    log("OK: dane")
except Exception as e:
    st.error("Problem z wczytaniem CSV.")
    st.exception(e); st.stop()

try:
    log("Wczytuję opisy klastrów…")
    cluster_info = load_cluster_info()
    log("OK: JSON")
except Exception as e:
    st.error("Problem z plikiem JSON z opisami klastrów.")
    st.exception(e); st.stop()

if df.empty:
    st.info("🔹 Baza jest pusta — zapisz przynajmniej jeden profil w panelu bocznym.")
    with st.expander("Diagnostyka"): st.write(diag)
    st.stop()

# --- Predykcja użytkownika ---
try:
    log("Predykcja użytkownika…")
    pred = predict_model(model, data=pd.DataFrame([user_data]))
    cluster_col_user = "Cluster" if "Cluster" in pred.columns else ("prediction_label" if "prediction_label" in pred.columns else None)
    if not cluster_col_user:
        st.error("Model nie zwrócił kolumny z etykietą klastra.")
        with st.expander("Diagnostyka"): st.write(diag)
        st.stop()
    pred_cluster_id = normalize_cluster_id(pred[cluster_col_user].values[0])
    log(f"OK: pred user → {pred_cluster_id}")
except Exception as e:
    st.error("Nie udało się policzyć predykcji dla profilu.")
    st.exception(e); 
    with st.expander("Diagnostyka"): st.write(diag)
    st.stop()

# --- Predykcja dla całej bazy i filtr grupy ---
try:
    log("Predykcja dla całej bazy…")
    all_pred = predict_model(model, data=df)
    cluster_col_all = "Cluster" if "Cluster" in all_pred.columns else "prediction_label"
    all_pred["_cluster_id_"] = all_pred[cluster_col_all].map(normalize_cluster_id)
    log("OK: pred all")
except Exception as e:
    st.error("Nie udało się policzyć predykcji dla bazy.")
    st.exception(e); 
    with st.expander("Diagnostyka"): st.write(diag)
    st.stop()

same_cluster_df = all_pred[all_pred["_cluster_id_"] == pred_cluster_id]

# Szybki health-check modelu (po wyliczeniu all_pred i same_cluster_df)
unikalne = sorted(all_pred["_cluster_id_"].unique().tolist())
st.info(f"🔎 Wykryte ID klastrów w modelu: {unikalne} (liczba: {len(unikalne)})")

# === KARTA KLASTRA ===
name, desc = cluster_name_desc(pred_cluster_id, cluster_info)
cluster_card(name, desc, pred_cluster_id, len(same_cluster_df))

# === ZAKŁADKI ===
tab_overview, tab_stats, tab_data = st.tabs(["🎯 Przegląd", "📊 Statystyki grupy", "📁 Dane"])

with tab_overview:
    with st.expander("🔍 Twój profil (podgląd)"):
        st.dataframe(pd.DataFrame([user_data]), hide_index=True, use_container_width=True)

    colA, colB, colC = st.columns(3)
    colA.metric("Osób w Twoim klastrze", len(same_cluster_df))
    colB.metric("Liczba wszystkich rekordów", len(df))
    colC.metric("Liczba klastrów w JSON", len(cluster_info))

    st.download_button(
        "⬇️ Pobierz swoją grupę (CSV)",
        data=same_cluster_df[COLS + ["_cluster_id_"]].to_csv(index=False).encode("utf-8"),
        file_name=f"twoj_klaster_{pred_cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with tab_stats:
    if same_cluster_df.empty:
        st.info("Brak danych do statystyk — dodaj więcej profili.")
    else:
        # === MEDIANA WIEKU (kategoria) ===
        col1, col2 = st.columns(2)
        med_grp = median_age_category(same_cluster_df)
        med_all = median_age_category(all_pred)
        col1.metric("Mediana wieku — Twoja grupa", med_grp)
        col2.metric("Mediana wieku — wszyscy", med_all)

        st.divider()

        # --- Pajęczynka: Twoja grupa vs. wszyscy ---
        st.markdown("### 🕸️ Profil grupy (radar)")
        grp_scores = radar_trait_scores(same_cluster_df)
        all_scores = radar_trait_scores(all_pred)
        radar_df = pd.DataFrame({
            "cecha": list(grp_scores.keys()),
            "Twoja grupa": list(grp_scores.values()),
            "Wszyscy": [all_scores[k] for k in grp_scores.keys()],
        })
        radar_long = radar_df.melt(id_vars="cecha", var_name="Zbiór", value_name="Wartość")
        fig_radar = px.line_polar(
            radar_long, r="Wartość", theta="cecha", color="Zbiór",
            line_close=True, range_r=[0,1], template=plotly_template,
            title="Radar: Twoja grupa vs. ogół"
        )
        fig_radar.update_traces(fill="toself", opacity=0.6)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # === ROZKŁAD PROCENTOWY: PŁEĆ ===
        st.markdown("### 👤 Płeć — rozkład procentowy")
        g1, g2 = st.columns(2)
        pct_gender_grp = percent_breakdown(same_cluster_df["gender"])
        pct_gender_all = percent_breakdown(all_pred["gender"])
        with g1:
            plot_percent_bar(pct_gender_grp, "Twoja grupa", plotly_template)
        with g2:
            plot_percent_bar(pct_gender_all, "Wszyscy", plotly_template)

        # === ROZKŁAD PROCENTOWY: EDUKACJA ===
        st.markdown("### 🎓 Wykształcenie — rozkład procentowy")
        e1, e2 = st.columns(2)
        pct_edu_grp = percent_breakdown(same_cluster_df["edu_level"])
        pct_edu_all = percent_breakdown(all_pred["edu_level"])
        with e1:
            plot_percent_bar(pct_edu_grp, "Twoja grupa", plotly_template)
        with e2:
            plot_percent_bar(pct_edu_all, "Wszyscy", plotly_template)

        # === ROZKŁAD PROCENTOWY: ULUBIONE MIEJSCA ===
        st.markdown("### 🗺️ Ulubione miejsca — rozkład procentowy")
        p1, p2 = st.columns(2)
        pct_place_grp = percent_breakdown(same_cluster_df["fav_place"])
        pct_place_all = percent_breakdown(all_pred["fav_place"])
        with p1:
            plot_percent_bar(pct_place_grp, "Twoja grupa", plotly_template)
        with p2:
            plot_percent_bar(pct_place_all, "Wszyscy", plotly_template)

with tab_data:
    st.write("### Podgląd wszystkich danych (bez etykiet modelu)")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.write("### Podgląd wszystkich danych z etykietami modelu")
    st.dataframe(all_pred[COLS + ["_cluster_id_"]], hide_index=True, use_container_width=True)

# === DIAGNOSTYKA (opcjonalna podglądówka) ===
with st.expander("Diagnostyka"):
    st.write(diag)

# === STOPKA ===
st.markdown(
    '<div class="footer">Made with ❤️ in Streamlit · Motyw: <b>'
    + st.session_state.theme +
    '</b></div>',
    unsafe_allow_html=True
)
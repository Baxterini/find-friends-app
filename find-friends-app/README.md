# find-friends-app
Aplikacja Streamlit do znajdowania „bliźniaczych” profili na podstawie krótkiej ankiety. Klasteryzacja (PyCaret), podgląd podobnych osób, wykresy i eksport wyników. Projekt z kursu rozwijany o własne ulepszenia.

# Find Friends — app 🫱🏻‍🫲🏻

Aplikacja **Streamlit** do znajdowania osób o podobnych cechach / preferencjach na podstawie krótkiej ankiety.  
W tle działa **klasteryzacja** (PyCaret), a użytkownik dostaje informację o swoim klastrze i listę osób najbardziej podobnych.

## Funkcje
- Formularz z kilkoma pytaniami (wiek, edukacja, ulubione rzeczy itp.)
- Przypisanie do klastra (model PyCaret)
- Podgląd osób z tego samego klastra + podstawowe wykresy
- Import/eksport danych CSV
- (Opcjonalnie) nazwy i opisy klastrów z pliku JSON

## Podgląd (lokalnie)
```bash
# 1) stwórz środowisko (conda lub venv – jak wolisz)
conda create -n findfriends python=3.11 -y
conda activate findfriends

# 2) zainstaluj zależności
pip install -r requirements.txt
# jeśli używasz pycaret: pip install "pycaret[full]"

# 3) uruchom Streamlit
streamlit run app.py

Wymagane pliki (domyślne nazwy)

    welcome_survey_simple_v2.csv – dane ankietowe (separator ;)

    welcome_survey_clustering_pipeline_v2 – wytrenowany pipeline PyCaret (ładowany przez load_model)

    welcome_survey_cluster_names_and_descriptions_v2.json – (opcjonalnie) słownik nazw i opisów klastrów

Struktura repo (propozycja)
.
├── app.py
├── requirements.txt
├── data/
│   └── welcome_survey_simple_v2.csv
├── models/
│   └── welcome_survey_clustering_pipeline_v2
├── config/
│   └── welcome_survey_cluster_names_and_descriptions_v2.json
├── README.md
└── .gitignore



Plik requirements.txt (start)
streamlit
pandas
plotly
pycaret
numpy


import pandas as pd
from pycaret.clustering import setup, create_model, assign_model, save_model

print("📂 Wczytywanie danych...")
df = pd.read_csv("welcome_survey_simple_v2.csv", sep=';')

categorical = ['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']

print("⚙️ Konfiguracja PyCaret...")
s = setup(
    data=df,
    normalize=True,
    session_id=123,
    categorical_features=categorical
)

print("🤖 Trening modelu KMeans...")
model = create_model('kmeans', num_clusters=8)

print("🏷️ Przypisywanie klastrów...")
clustered_df = assign_model(model)

print("💾 Zapis modelu...")
save_model(model, "welcome_survey_clustering_pipeline_v2")

print("✅ Gotowe! Model zapisany.")

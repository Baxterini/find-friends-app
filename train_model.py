import pandas as pd
from pycaret.clustering import setup, create_model, assign_model, save_model, load_model, predict_model
import sklearn, pycaret, numpy

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

print("📦 Wersje bibliotek:")
print("scikit-learn:", sklearn.__version__)
print("pycaret:", pycaret.__version__)
print("numpy:", numpy.__version__)

print("🔍 Test ładowania modelu...")
loaded = load_model("welcome_survey_clustering_pipeline_v2")
sample = df.iloc[[0]]
print("🔮 Predykcja testowa:", predict_model(loaded, data=sample))

print("✅ Gotowe! Model zapisany i sprawdzony.")

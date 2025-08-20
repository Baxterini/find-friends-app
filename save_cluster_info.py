import json
from pathlib import Path

# --- Ścieżka docelowa ---
OUTPUT_PATH = Path(__file__).parent / "welcome_survey_cluster_names_and_descriptions_v2.json"

# --- Nazwy i opisy klastrów ---
cluster_info = {
    "0": {
        "name": "🌿 Introwertyczni Marzyciele",
        "description": "Osoby ceniące ciszę, naturę i głębokie rozmowy. Najczęściej spotkasz ich w lesie z książką lub kubkiem herbaty."
    },
    "1": {
        "name": "🎉 Społeczni Entuzjaści",
        "description": "Uwielbiają towarzystwo, imprezy i nowe znajomości. Są energiczni, rozmowni i kochają psy!"
    },
    "2": {
        "name": "🧘‍♂️ Harmoniści",
        "description": "Szukają równowagi we wszystkim. Lubią wodę, jogę, medytację i dobrze zaparzoną kawę."
    },
    "3": {
        "name": "🔬 Umysły Analityczne",
        "description": "Myślą logicznie, często związani z technologią lub nauką. Cenią konkret i jasną komunikację."
    },
    "4": {
        "name": "🎨 Artyści i Twórcy",
        "description": "Z kreatywną duszą, kochają muzykę, sztukę, wyrażanie siebie. W grupie to często osoby inspirujące."
    },
    "5": {
        "name": "🌍 Poszukiwacze Przygód",
        "description": "Eksplorują nowe miejsca, kultury i kuchnie. Kochają góry i podróże bez planu."
    },
    "6": {
        "name": "🛋️ Domatorzy",
        "description": "Najlepiej czują się w swoim przytulnym kącie. Cenią stabilność, ciepło i dobry serial."
    },
    "7": {
        "name": "💼 Ambitni Realizatorzy",
        "description": "Zorientowani na cele, rozwój i sukces. Zawsze z planem w ręku i kolejnym kursem na horyzoncie."
    }
}

# --- Zapis pliku JSON ---
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cluster_info, f, ensure_ascii=False, indent=4)

print(f"✅ Plik JSON zapisany w: {OUTPUT_PATH}")
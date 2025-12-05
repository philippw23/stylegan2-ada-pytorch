

from pathlib import Path

json_path = Path(__file__).resolve().parent / "dataset" / "BTXRD_resized_sorted" /"dataset.json"

with open(json_path, "r", encoding="utf-8") as f:
    text = f.read()

kommas = text.count(",")
print("Anzahl der Kommas:", kommas)

import joblib
import json
import os

MODEL_PATH = "models/isolation_forest.pkl"
OUTPUT_PATH = "models/feature_columns.json"

print("Checking model path:", os.path.abspath(MODEL_PATH))

model = joblib.load(MODEL_PATH)

print("\nExtracted n_features_in_:", model.n_features_in_)
print("Extracting real feature names...")

features = list(model.feature_names_in_)
print("Extracted:", len(features))

with open(OUTPUT_PATH, "w") as f:
    json.dump(features, f, indent=2)

print("\n[✓] Saved REAL features to:", OUTPUT_PATH)

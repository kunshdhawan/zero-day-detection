import pandas as pd
import joblib
import json
import numpy as np
import os
import gradio as gr

MODEL_PATH = "models/isolation_forest.pkl"
FEATURES_PATH = "models/feature_columns.json"

# Load model
model = joblib.load(MODEL_PATH)

# Load features
with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)

EXPECTED_FEATURES = len(FEATURES)

def preprocess(df):
    # Ensure correct columns
    df = df.reindex(columns=FEATURES, fill_value=0)
    return df

def predict(csv_file):
    try:
        df = pd.read_csv(csv_file.name, low_memory=False)

        df = preprocess(df)

        if df.shape[1] != EXPECTED_FEATURES:
            return f"ERROR: CSV has {df.shape[1]} features, expected {EXPECTED_FEATURES}"

        preds = model.predict(df)

        results = ["ANOMALY" if p == -1 else "NORMAL" for p in preds]

        output = "\n".join(results[:20])
        if len(results) > 20:
            output += "\n...\n(Showing first 20)"

        return output

    except Exception as e:
        return f"ERROR:\n{str(e)}"

app = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload CSV"),
    outputs=gr.Textbox(label="Results"),
    title="Zero-Day Anomaly Detector (Trained on 103 Features)",
    description="Uploads CSV and aligns it to the exact 103 training features."
)

if __name__ == "__main__":
    app.launch()

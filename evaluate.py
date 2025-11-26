import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt


MODEL_PATH = "models/isolation_forest.pkl"
FEATURES_PATH = "models/feature_columns.json"
CSV_PATH = "data/processed/combined.csv"
RESULT_DIR = "results"
PREDICTION_OUTPUT = f"{RESULT_DIR}/evaluation_predictions.csv"


def load_model_and_features():
    print("[+] Loading model...")
    model = joblib.load(MODEL_PATH)

    print("[+] Loading feature columns...")
    with open(FEATURES_PATH) as f:
        features = json.load(f)

    os.makedirs(RESULT_DIR, exist_ok=True)
    return model, features


def preprocess(df, features):
    df = df.reindex(columns=features, fill_value=0)
    return df


def save_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/confusion_matrix.png")
    plt.close()


def save_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"{RESULT_DIR}/roc_curve.png")
    plt.close()


def save_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"{RESULT_DIR}/precision_recall_curve.png")
    plt.close()


def save_anomaly_score_hist(scores):
    plt.figure()
    plt.hist(scores, bins=40, color='gray')
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{RESULT_DIR}/anomaly_score_hist.png")
    plt.close()


def evaluate(csv_path):
    model, features = load_model_and_features()

    print(f"[+] Loading CSV: {csv_path}")
    print("[+] Loading CSV safely...")
    try:
        df = pd.read_csv(
        csv_path,
        engine="python",
        on_bad_lines="skip"
    )
    except Exception as e:
        print("Python engine failed, trying C engine...")
        df = pd.read_csv(
        csv_path,
        engine="c",
        on_bad_lines="skip"
    )



    has_label = "Label" in df.columns

    if has_label:
        print("[+] Detected 'Label' column. Full evaluation enabled.")
        y_true = df["Label"].apply(lambda x: 1 if x == "BENIGN" else -1)
        df = df.drop(columns=["Label"])
    else:
        print("[!] No label column found. Evaluation will only produce predictions.")
        y_true = None

    print("[+] Preprocessing...")
    df_clean = preprocess(df, features)

    if df_clean.shape[1] != len(features):
        return f"CSV has {df_clean.shape[1]} columns but model expects {len(features)}."

    print("[+] Running predictions...")
    predictions = model.predict(df_clean)
    anomaly_scores = model.decision_function(df_clean)

    # Save predictions
    df_output = pd.DataFrame({
        "Prediction": ["NORMAL" if p == 1 else "ANOMALY" for p in predictions],
        "Anomaly_Score": anomaly_scores
    })
    df_output.to_csv(PREDICTION_OUTPUT, index=False)
    print(f"[✓] Saved predictions → {PREDICTION_OUTPUT}")

    # Save score histogram
    save_anomaly_score_hist(anomaly_scores)

    if y_true is None:
        return "[✓] Prediction-only evaluation complete."

    print("[+] Computing metrics...")
    print("\n[Classification Report]")
    print(classification_report(y_true, predictions, target_names=["ANOMALY", "NORMAL"]))

    cm = confusion_matrix(y_true, predictions)
    print("\n[Confusion Matrix]")
    print(cm)
    save_confusion_matrix(cm, ["ANOMALY", "NORMAL"])

    # ROC Curve
    try:
        roc = roc_auc_score(y_true, anomaly_scores)
        print(f"\n[ROC AUC Score] {roc:.4f}")
        save_roc_curve(y_true, anomaly_scores)
    except Exception:
        print("[!] Could not compute ROC curve (possibly only one class present).")

    # Precision–Recall curve
    save_precision_recall(y_true, anomaly_scores)

    print(f"\n[✓] All plots saved in '{RESULT_DIR}/'")
    return "[✓] Full evaluation complete."


if __name__ == "__main__":
    print(evaluate(CSV_PATH))
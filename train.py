import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

BENIGN_PATH = "data/processed/benign_only.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_PATH = "models/isolation_forest.pkl"


def load_benign_dataset():
    print("[+] Loading benign-only dataset...")
    df = pd.read_csv(BENIGN_PATH, low_memory=False)
    print("[+] Loaded shape:", df.shape)
    return df


def prepare_training_data(df):
    print("[+] Preparing training data...")
    X_train = df.drop(columns=["Label"], errors="ignore")
    print("[+] Training features:", X_train.shape)
    return X_train


def train_isolation_forest(X_train):
    print("[+] Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    print("[✓] Model training complete.")
    return model


def save_model(model):
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"[✓] Model saved at: {MODEL_OUTPUT_PATH}")


def main():
    print("\n=== ZERO-DAY MODEL TRAINING ===\n")

    df = load_benign_dataset()
    X_train = prepare_training_data(df)
    model = train_isolation_forest(X_train)
    save_model(model)

    print("\n=== TRAINING COMPLETE ===\n")


if __name__ == "__main__":
    main()

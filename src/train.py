"""
train.py (Path A - baseline)
Trains a baseline Random Forest to classify benign vs attack traffic
using the balanced dataset built by preprocessing.py

NOTE: The more advanced federated/autoencoder version lives in
train_advanced.py - this is deliberately the simple version first.
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = os.path.join("data", "processed", "binary_balanced.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_rf.pkl")

# Columns to exclude from features (labels, not measurements)
LABEL_COLS = ["label", "binary_label"]


def train_baseline():
    print("Loading balanced dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=LABEL_COLS)
    y = df["binary_label"]

    print(f"Feature matrix: {X.shape}, Target: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save the feature column order - the dashboard/simulation will need this later
    feature_cols_path = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_cols_path, "w") as f:
        f.write("\n".join(X.columns.tolist()))
    print(f"Feature columns saved to {feature_cols_path}")


if __name__ == "__main__":
    train_baseline()
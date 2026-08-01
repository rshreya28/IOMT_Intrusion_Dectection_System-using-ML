"""
alert_engine.py
Takes simulated device traffic (from simulate_traffic.py), runs each row
through the trained baseline model, and generates structured alerts for
anything flagged as an attack.
"""

import pandas as pd
import joblib
import os
from datetime import datetime

from simulate_traffic import stream_traffic

MODEL_PATH = os.path.join("models", "baseline_rf.pkl")
FEATURE_COLS_PATH = os.path.join("models", "feature_columns.txt")

# Columns present in the raw data that are NOT model features (labels)
NON_FEATURE_COLS = ["label", "binary_label"]


def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = f.read().splitlines()
    return model, feature_cols


def row_to_features(row, feature_cols):
    """Extract just the model's expected feature columns from a raw data row,
    in the correct order, as a single-row DataFrame ready for prediction."""
    values = {col: row[col] for col in feature_cols}
    return pd.DataFrame([values])


def generate_alerts(sample_size=100, delay=0.0):
    """
    Runs simulated traffic through the model one row at a time.
    Yields a structured alert dict for EVERY event (both normal and suspicious),
    so the dashboard can show live status for all devices, not just attacks.
    """
    model, feature_cols = load_model_and_features()

    for device, row in stream_traffic(sample_size=sample_size, delay=delay):
        X = row_to_features(row, feature_cols)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = max(proba)

        true_label = row.get("label", "unknown")  # for our own testing only

        alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "status": "suspicious" if prediction == "attack" else "normal",
            "confidence": round(float(confidence), 4),
            "true_label": true_label,  # remove/hide this in the final dashboard demo
        }
        yield alert


if __name__ == "__main__":
    print("Running alert engine on simulated traffic (first 15 events)...\n")
    for i, alert in enumerate(generate_alerts(sample_size=15)):
        flag = "🔴" if alert["status"] == "suspicious" else "🟢"
        print(f"{flag} [{alert['timestamp']}] {alert['device']:35s} "
              f"| Status: {alert['status']:10s} | Confidence: {alert['confidence']:.2%} "
              f"| (true: {alert['true_label']})")
"""
alert_engine_advanced.py
Same purpose as alert_engine.py, but uses the federated autoencoder model
(Path B) instead of the baseline Random Forest (Path A).

Flags traffic as suspicious when reconstruction error exceeds the saved
95th-percentile threshold, and explains WHY using SHAP.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
from datetime import datetime

from simulate_traffic import stream_traffic
from model import LightIDSAutoencoder
from explain import AlertExplainer

MODEL_PATH = os.path.join("models", "federated_autoencoder.pt")
SCALER_PATH = os.path.join("models", "federated_scaler.pkl")
FEATURE_NAMES_PATH = os.path.join("models", "federated_feature_names.txt")
THRESHOLD_PATH = os.path.join("models", "federated_threshold.txt")

NON_FEATURE_COLS = ["label", "binary_label"]


def load_advanced_artifacts():
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = f.read().splitlines()
    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())

    scaler = joblib.load(SCALER_PATH)

    model = LightIDSAutoencoder(input_dim=len(feature_names))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    return model, scaler, feature_names, threshold


def row_to_scaled_tensor(row, feature_names, scaler):
    values = np.array([[row[col] for col in feature_names]], dtype=np.float32)
    scaled = scaler.transform(values)
    return torch.tensor(scaled, dtype=torch.float32), scaled[0]


def generate_alerts_advanced(sample_size=100, delay=0.0, explain_every=1):
    """
    Same interface as alert_engine.generate_alerts(), but powered by the
    federated autoencoder. Yields one alert dict per event.

    explain_every: only run SHAP (slow) every Nth suspicious event, to keep
    the dashboard responsive. Set to 1 to explain every alert.
    """
    model, scaler, feature_names, threshold = load_advanced_artifacts()

    # Small background sample for SHAP (kept small - SHAP is slow)
    background = np.random.RandomState(42).randn(50, len(feature_names)).astype(np.float32)
    explainer = AlertExplainer(model, background, feature_names, top_k=3)

    suspicious_count = 0

    for device, row in stream_traffic(sample_size=sample_size, delay=delay):
        x_tensor, x_scaled = row_to_scaled_tensor(row, feature_names, scaler)
        error = model.reconstruction_error(x_tensor).item()
        is_suspicious = error > threshold

        reason = None
        if is_suspicious:
            suspicious_count += 1
            if suspicious_count % explain_every == 0:
                reason = explainer.format_alert(x_scaled)

        true_label = row.get("label", "unknown")

        alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "status": "suspicious" if is_suspicious else "normal",
            "reconstruction_error": round(float(error), 4),
            "reason": reason or "",
            "true_label": true_label,
        }
        yield alert


if __name__ == "__main__":
    print("Running ADVANCED (federated autoencoder) alert engine on simulated traffic...\n")
    for i, alert in enumerate(generate_alerts_advanced(sample_size=15)):
        flag = "🔴" if alert["status"] == "suspicious" else "🟢"
        print(f"{flag} [{alert['timestamp']}] {alert['device']:35s} "
              f"| Status: {alert['status']:10s} | Error: {alert['reconstruction_error']:.4f} "
              f"| (true: {alert['true_label']})")
        if alert["reason"]:
            print(f"     {alert['reason']}")
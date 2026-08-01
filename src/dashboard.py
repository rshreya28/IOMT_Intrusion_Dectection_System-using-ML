"""
dashboard.py
Live-updating Streamlit dashboard for the MIoT Intrusion Detection System.
Supports switching between:
  - Path A: Baseline Random Forest (fast, simple, binary classifier)
  - Path B: Federated Autoencoder (privacy-preserving, anomaly-based,
            explainable via SHAP, drift-aware)

Run with: streamlit run src/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

from alert_engine import generate_alerts
from alert_engine_advanced import generate_alerts_advanced

st.set_page_config(page_title="MIoT-IDS Live Dashboard", layout="wide")

st.title("🏥 Medical IoT — Intrusion Detection Dashboard")
st.caption("Live simulated device traffic, classified in real time.")

# --- Controls ---
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    model_choice = st.selectbox(
        "Detection model",
        ["Path A: Baseline (Random Forest)", "Path B: Federated Autoencoder"],
    )
with col_b:
    sample_size = st.slider("Number of events to simulate", 20, 500, 100, step=20)
with col_c:
    delay = st.slider("Delay between events (seconds)", 0.0, 1.0, 0.1, step=0.05)
with col_d:
    start = st.button("▶ Start Simulation", type="primary")

st.divider()

# --- Layout placeholders ---
status_placeholder = st.empty()
chart_placeholder = st.empty()
feed_placeholder = st.empty()

if start:
    device_status = {}
    alert_log = []
    attack_type_counts = {}

    is_advanced = model_choice.startswith("Path B")

    if is_advanced:
        alert_stream = generate_alerts_advanced(sample_size=sample_size, delay=delay)
    else:
        alert_stream = generate_alerts(sample_size=sample_size, delay=delay)

    for alert in alert_stream:
        device_status[alert["device"]] = alert["status"]
        alert_log.insert(0, alert)

        if alert["status"] == "suspicious":
            attack_type_counts[alert["true_label"]] = attack_type_counts.get(alert["true_label"], 0) + 1

        # --- Device status panel ---
        with status_placeholder.container():
            st.subheader("📟 Device Fleet Status")
            cols = st.columns(4)
            for i, (device, status) in enumerate(device_status.items()):
                icon = "🔴" if status == "suspicious" else "🟢"
                with cols[i % 4]:
                    st.metric(label=device, value=f"{icon} {status.upper()}")

        # --- Attack type chart ---
        with chart_placeholder.container():
            if attack_type_counts:
                st.subheader("📊 Detected Attack Types (this session)")
                chart_df = pd.DataFrame(
                    list(attack_type_counts.items()), columns=["Attack Type", "Count"]
                ).sort_values("Count", ascending=False)
                st.bar_chart(chart_df.set_index("Attack Type"))

        # --- Live alert feed ---
        with feed_placeholder.container():
            st.subheader("🚨 Live Alert Feed")
            feed_df = pd.DataFrame(alert_log[:30])

            if is_advanced:
                display_cols = ["timestamp", "device", "status",
                                 "reconstruction_error", "reason", "true_label"]
            else:
                display_cols = ["timestamp", "device", "status",
                                 "confidence", "true_label"]

            display_cols = [c for c in display_cols if c in feed_df.columns]
            st.dataframe(feed_df[display_cols], use_container_width=True, hide_index=True)

    st.success(f"Simulation complete — {model_choice}")
else:
    st.info("Choose a model and settings above, then click **Start Simulation**.")
    st.markdown("""
    **Path A — Baseline (Random Forest):** fast, supervised, trained centrally 
    on all data. Simple and highly accurate on this dataset, but requires 
    pooling raw data in one place.

    **Path B — Federated Autoencoder:** trains across 5 simulated hospitals 
    without sharing raw data (only model weights), learns only from *normal* 
    traffic, flags anomalies via reconstruction error, and explains each 
    alert with SHAP. Slower per-alert (SHAP explanation), but privacy-preserving 
    and more aligned with real healthcare data-sharing constraints.
    """)
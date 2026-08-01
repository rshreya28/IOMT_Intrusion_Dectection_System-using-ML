"""
dashboard.py
Live-updating Streamlit dashboard for the MIoT Intrusion Detection System.
Shows simulated device traffic being classified in real time, with a
device status panel and a scrolling alert feed.

Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import time

from alert_engine import generate_alerts

st.set_page_config(page_title="MIoT-IDS Live Dashboard", layout="wide")

st.title("🏥 Medical IoT — Intrusion Detection Dashboard")
st.caption("Live simulated device traffic, classified in real time by the baseline model.")

# --- Controls ---
col_a, col_b, col_c = st.columns(3)
with col_a:
    sample_size = st.slider("Number of events to simulate", 20, 500, 100, step=20)
with col_b:
    delay = st.slider("Delay between events (seconds)", 0.0, 1.0, 0.1, step=0.05)
with col_c:
    start = st.button("▶ Start Simulation", type="primary")

# --- Layout placeholders ---
status_placeholder = st.empty()
chart_placeholder = st.empty()
feed_placeholder = st.empty()

if start:
    device_status = {}   # device -> latest status
    alert_log = []        # running list of alerts, most recent first
    attack_type_counts = {}

    for alert in generate_alerts(sample_size=sample_size, delay=delay):
        device_status[alert["device"]] = alert["status"]
        alert_log.insert(0, alert)  # newest first

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
            feed_df = pd.DataFrame(alert_log[:30])  # show last 30 events
            st.dataframe(feed_df, use_container_width=True, hide_index=True)

    st.success("Simulation complete.")
else:
    st.info("Set your simulation options above and click **Start Simulation**.")
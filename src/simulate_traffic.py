"""
simulate_traffic.py
Simulates a live stream of medical IoT device traffic by replaying rows
from test_iomt.csv one at a time, each tagged with a fake device name.

This feeds the alert engine (alert_engine.py) which will use the trained
model to flag suspicious activity per device.
"""

import pandas as pd
import random
import time
import os

TEST_PATH = os.path.join("data", "raw", "test_iomt.csv")

# Fake medical device fleet - realistic IoMT device names for the demo
DEVICE_NAMES = [
    "ECG Monitor - Room 101",
    "Infusion Pump - Room 102",
    "SpO2 Sensor - Room 103",
    "Patient Vitals Hub - ICU-1",
    "Insulin Pump - Room 104",
    "EEG Monitor - Room 105",
    "Smart Bed Sensor - Room 106",
    "Blood Pressure Monitor - Room 107",
]


def stream_traffic(sample_size=2000, delay=0.0, seed=42):
    """
    Yields (device_name, row_as_series) one at a time, simulating
    live incoming traffic from a fleet of medical devices.

    sample_size: how many rows to pull from the test file for this simulation
    delay: seconds to wait between rows (0 = as fast as possible, useful for
           testing; set higher, e.g. 0.05-0.5, for a more realistic live demo)
    """
    random.seed(seed)

    # Read a chunk of test data - random sample so we get a mix of attack/benign
    df = pd.read_csv(TEST_PATH)
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)

    for _, row in df_sample.iterrows():
        device = random.choice(DEVICE_NAMES)
        yield device, row
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    # Quick manual test: print the first 10 simulated events
    print("Simulating live traffic (first 10 events)...\n")
    for i, (device, row) in enumerate(stream_traffic(sample_size=10)):
        true_label = row.get("label", "unknown")
        print(f"[{i+1}] Device: {device:35s} | True label: {true_label}")
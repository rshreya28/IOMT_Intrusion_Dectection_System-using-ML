"""
data_loader.py
Loads the CICIoMT2024 dataset and prepares it for both:
  - simple exploration (load_sample, load_full - Path A)
  - federated anomaly-detection training (make_client_loaders - Path B)
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TRAIN_PATH = os.path.join("data", "raw", "train_iomt.csv")
TEST_PATH = os.path.join("data", "raw", "test_iomt.csv")
BALANCED_PATH = os.path.join("data", "processed", "binary_balanced.csv")

LABEL_COLS = ["label", "binary_label"]


# ---------- Path A helpers (kept for compatibility) ----------

def load_sample(path=TRAIN_PATH, n_rows=5):
    df = pd.read_csv(path, nrows=n_rows)
    print(f"\n--- Sample from {path} ---")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    return df


def load_full(path=TRAIN_PATH):
    print(f"Loading full dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


# ---------- Path B: federated client data ----------

def make_client_loaders(config):
    """
    Loads the balanced dataset, scales features, and splits it into
    `num_clients` simulated hospitals/devices.

    For each client:
      - X_train: ONLY benign traffic (the autoencoder learns "normal" from this)
      - X_test / y_test: a held-out mix of benign + attack traffic, used to
        evaluate how well the model catches anomalies

    Returns:
        client_data: list of dicts, one per client, each with
                     X_train, X_test, y_test (numpy arrays)
        scaler: the fitted StandardScaler (needed later to scale any new
                live traffic the same way before feeding it to the model)
        feature_cols: list of real feature column names, in order,
                      matching the columns of X_train/X_test
    """
    print(f"Loading {BALANCED_PATH} for federated client split...")
    df = pd.read_csv(BALANCED_PATH)

    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].astype(str).to_numpy()  # force plain numpy array (avoids pandas 3.x arrow-string indexing issues)

    # Scale ALL features together first (fit once, globally) - this simulates
    # a shared, agreed-upon preprocessing standard across hospitals, which is
    # realistic (they'd agree on units/normalization even if data stays local)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    num_clients = config["num_clients"]
    rng = np.random.default_rng(seed=42)
    shuffled_idx = rng.permutation(len(X_scaled))
    client_index_chunks = np.array_split(shuffled_idx, num_clients)

    client_data = []
    for i, idx_chunk in enumerate(client_index_chunks):
        X_client = X_scaled[idx_chunk]
        y_client = y[idx_chunk]

        benign_mask = y_client == "benign"
        X_benign = X_client[benign_mask]
        y_benign = y_client[benign_mask]
        X_attack = X_client[~benign_mask]
        y_attack = y_client[~benign_mask]

        # Split benign into train (80%) and a portion held out for testing
        X_benign_train, X_benign_test, y_benign_train, y_benign_test = train_test_split(
            X_benign, y_benign, test_size=0.2, random_state=42
        )

        # Test set = held-out benign + all this client's attack traffic
        X_test = np.concatenate([X_benign_test, X_attack])
        y_test = np.concatenate([y_benign_test, y_attack])

        client_data.append({
            "client_id": i,
            "X_train": X_benign_train,   # benign-only, for autoencoder training
            "X_test": X_test,             # mixed, for evaluation
            "y_test": y_test,
        })

        print(f"  Client {i}: train(benign)={len(X_benign_train)}, "
              f"test(mixed)={len(X_test)} "
              f"(benign={len(X_benign_test)}, attack={len(X_attack)})")

    return client_data, scaler, feature_cols


if __name__ == "__main__":
    sample = load_sample()
    print(sample.head())
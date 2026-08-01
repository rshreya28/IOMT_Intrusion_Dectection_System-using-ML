"""
preprocessing.py
Builds a balanced binary (benign vs attack) dataset from CICIoMT2024,
ensuring ALL 17 attack types are represented - not just whichever ones
happen to appear first in the file.

The raw file is sorted in large blocks by label (all DDoS UDP rows together,
then all DDoS ICMP, etc.) - reading only the first N rows would only ever
capture the first few attack types. This version scans the FULL file once,
buckets rows by their real label, then samples a capped amount from each
bucket so common attacks (DDoS UDP: 1.6M rows) don't drown out rare ones
(Ping Sweep: 740 rows).
"""

import pandas as pd
import os

TRAIN_PATH = os.path.join("data", "raw", "train_iomt.csv")
OUTPUT_PATH = os.path.join("data", "processed", "binary_balanced.csv")

CHUNK_SIZE = 500_000
CAP_PER_ATTACK_TYPE = 15_000   # max rows to keep per individual attack type
                                 # (rare types with fewer than this just keep everything)
BENIGN_TARGET = 190_000         # keep roughly all benign rows


def build_balanced_binary_dataset():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Buckets: one list of DataFrame chunks per label
    label_buckets = {}
    benign_rows = []
    benign_count = 0

    reader = pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader):
        chunk["binary_label"] = chunk["label"].apply(
            lambda x: "benign" if x.strip().lower() == "benign" else "attack"
        )

        # --- benign: collect up to target, same as before ---
        benign_chunk = chunk[chunk["binary_label"] == "benign"]
        if benign_count < BENIGN_TARGET and len(benign_chunk) > 0:
            benign_rows.append(benign_chunk)
            benign_count += len(benign_chunk)

        # --- attacks: bucket by REAL label (e.g. "Port Scan", "ARP Spoofing") ---
        attack_chunk = chunk[chunk["binary_label"] == "attack"]
        for label_name, group in attack_chunk.groupby("label"):
            existing = label_buckets.get(label_name)
            current_count = sum(len(df) for df in existing) if existing else 0

            if current_count >= CAP_PER_ATTACK_TYPE:
                continue  # already have enough of this attack type

            needed = CAP_PER_ATTACK_TYPE - current_count
            to_add = group.sample(min(needed, len(group)), random_state=42)

            label_buckets.setdefault(label_name, []).append(to_add)

        print(f"Chunk {i+1}: benign so far={benign_count}, "
              f"attack types seen so far={len(label_buckets)}")

    # --- Combine everything ---
    attack_dfs = []
    print("\nFinal per-attack-type counts:")
    for label_name, chunks in label_buckets.items():
        combined = pd.concat(chunks, ignore_index=True)
        print(f"  {label_name}: {len(combined)}")
        attack_dfs.append(combined)

    df_balanced = pd.concat(benign_rows + attack_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    df_balanced.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved balanced dataset: {df_balanced.shape}")
    print("\nFinal binary_label distribution:")
    print(df_balanced["binary_label"].value_counts())
    print("\nFinal original-label distribution:")
    print(df_balanced["label"].value_counts())


if __name__ == "__main__":
    build_balanced_binary_dataset()
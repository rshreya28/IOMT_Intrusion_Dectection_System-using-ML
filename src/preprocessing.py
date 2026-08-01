"""
preprocessing.py
Builds a balanced binary (benign vs attack) sample from CICIoMT2024
for a first baseline model.
"""

import pandas as pd
import os

TRAIN_PATH = os.path.join("data", "raw", "train_iomt.csv")
OUTPUT_PATH = os.path.join("data", "processed", "binary_balanced.csv")

CHUNK_SIZE = 500_000
BENIGN_TARGET = 190_000   # roughly all benign rows available
ATTACK_TARGET = 190_000   # sample this many attack rows, to match

def build_balanced_binary_dataset():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    benign_rows = []
    attack_rows = []
    benign_count = 0
    attack_count = 0

    reader = pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader):
        chunk["binary_label"] = chunk["label"].apply(
            lambda x: "benign" if x.strip().lower() == "benign" else "attack"
        )

        benign_chunk = chunk[chunk["binary_label"] == "benign"]
        attack_chunk = chunk[chunk["binary_label"] == "attack"]

        if benign_count < BENIGN_TARGET:
            benign_rows.append(benign_chunk)
            benign_count += len(benign_chunk)

        if attack_count < ATTACK_TARGET:
            # only take a slice so we don't overshoot
            needed = ATTACK_TARGET - attack_count
            attack_rows.append(attack_chunk.sample(min(needed, len(attack_chunk)), random_state=42))
            attack_count += min(needed, len(attack_chunk))

        print(f"Chunk {i+1}: benign so far={benign_count}, attack so far={attack_count}")

        if benign_count >= BENIGN_TARGET and attack_count >= ATTACK_TARGET:
            break

    df_balanced = pd.concat(benign_rows + attack_rows, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    df_balanced.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved balanced dataset: {df_balanced.shape}")
    print(df_balanced["binary_label"].value_counts())

if __name__ == "__main__":
    build_balanced_binary_dataset()
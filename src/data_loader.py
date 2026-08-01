"""
data_loader.py
Loads the CICIoMT2024 dataset (train/test CSVs) for the MIoT IDS project.
"""

import pandas as pd
import os

# Paths to the dataset files
TRAIN_PATH = os.path.join("data", "raw", "train_iomt.csv")
TEST_PATH = os.path.join("data", "raw", "test_iomt.csv")


def load_sample(path=TRAIN_PATH, n_rows=5):
    """Quick peek at the data structure without loading the full file."""
    df = pd.read_csv(path, nrows=n_rows)
    print(f"\n--- Sample from {path} ---")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    return df


def load_full(path=TRAIN_PATH):
    """Load the full dataset. Use only when you actually need all rows —
    these files are large (train ~1.7GB), so this can take a while and use significant RAM."""
    print(f"Loading full dataset from {path} ... this may take a minute.")
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def get_label_column(df):
    """Try to auto-detect the label/attack column by common naming patterns."""
    candidates = [c for c in df.columns if c.lower() in
                  ["label", "labels", "attack", "attack_type", "class", "type"]]
    if candidates:
        print(f"Likely label column(s): {candidates}")
        return candidates[0]
    else:
        print("Could not auto-detect label column. Check df.columns manually.")
        return None


if __name__ == "__main__":
    sample = load_sample()
    label_col = get_label_column(sample)
    if label_col:
        print(f"\nSample value counts for '{label_col}':")
        print(sample[label_col].value_counts())
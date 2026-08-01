"""
Entry point. Ties together:
  1. Federated training of a lightweight anomaly-detection autoencoder
  2. Per-client drift monitoring on live/streaming test data
  3. Explainable alerts for flagged samples

Run:  python src/train.py --config ../config.yaml
"""

import argparse
import yaml
import numpy as np
import torch

from data_loader import make_client_loaders
from federated import run_federated_training
from drift import DriftMonitor
from explain import AlertExplainer


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("== Loading and partitioning data across simulated clients ==")
    client_data, scaler = make_client_loaders(config)

    # Infer input_dim from the actual data rather than trusting the config,
    # so this doesn't silently break if the dataset's feature count differs.
    actual_dim = client_data[0]["X_train"].shape[1]
    if actual_dim != config["input_dim"]:
        print(f"[info] config input_dim={config['input_dim']} != actual "
              f"features={actual_dim}; using {actual_dim}")
        config["input_dim"] = actual_dim

    print("\n== Federated training ==")
    global_model = run_federated_training(client_data, config)

    print("\n== Setting up per-client drift monitors + explainers ==")
    feature_names = [f"feature_{i}" for i in range(config["input_dim"])]  # replace with real names

    # Pick a threshold from normal-only training error (simple heuristic:
    # mean + 3*std of reconstruction error on training data)
    all_train = np.concatenate([c["X_train"] for c in client_data])
    train_errors = global_model.reconstruction_error(
        torch.tensor(all_train, dtype=torch.float32)
    ).numpy()
    threshold = train_errors.mean() + 3 * train_errors.std()
    print(f"Anomaly threshold (mean + 3*std of training error): {threshold:.4f}")

    for i, client in enumerate(client_data):
        print(f"\n--- Client {i} ---")
        monitor = DriftMonitor(delta=config["drift_delta"])
        explainer = AlertExplainer(
            global_model, client["X_train"], feature_names,
            top_k=config["top_k_reasons"],
        )

        X_test = client["X_test"]
        errors = global_model.reconstruction_error(
            torch.tensor(X_test, dtype=torch.float32)
        ).numpy()

        n_alerts = 0
        for j, err in enumerate(errors):
            drifted = monitor.update(err)
            if drifted:
                print(f"  [drift] distribution shift detected at sample {j} "
                      f"-> consider retraining this client")
            if err > threshold:
                n_alerts += 1
                if n_alerts <= 2:  # only print a couple of example explanations
                    print(f"  {explainer.format_alert(X_test[j])}")

        print(f"  Total alerts: {n_alerts}/{len(X_test)} test samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)

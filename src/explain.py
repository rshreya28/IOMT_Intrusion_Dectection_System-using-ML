"""
Turns a raw "reconstruction error is high" alert into a human-readable reason,
e.g. "flagged mainly due to: packet_interarrival_time, dst_port_entropy".

Uses SHAP's KernelExplainer on the per-feature squared reconstruction error
(how much each input feature contributed to the model failing to reconstruct
it) rather than on the model's raw output -- this gives a much more intuitive
"which feature looked weird" explanation for an anomaly-detection model.
"""

import numpy as np
import shap
import torch


class AlertExplainer:
    def __init__(self, model, background_X, feature_names, top_k=3):
        self.model = model
        self.feature_names = feature_names
        self.top_k = top_k

        def predict_fn(X):
            x_t = torch.tensor(X, dtype=torch.float32)
            return self.model.reconstruction_error(x_t).numpy()

        # KernelExplainer works model-agnostically -- fine for a small model,
        # swap for DeepExplainer if speed becomes an issue at scale.
        self.explainer = shap.KernelExplainer(predict_fn, background_X[:50])

    def explain(self, x_row):
        """x_row: 1D numpy array (single traffic sample)."""
        shap_values = self.explainer.shap_values(x_row.reshape(1, -1), nsamples=100)
        contributions = np.abs(shap_values[0])
        top_idx = np.argsort(contributions)[::-1][: self.top_k]
        reasons = [(self.feature_names[i], float(contributions[i])) for i in top_idx]
        return reasons

    def format_alert(self, x_row):
        reasons = self.explain(x_row)
        reason_str = ", ".join(f"{name} (impact {val:.3f})" for name, val in reasons)
        return f"ALERT — flagged mainly due to: {reason_str}"

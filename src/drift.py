"""
Wraps River's ADWIN drift detector around the model's reconstruction error
stream. When traffic distribution shifts (new device firmware, new attack
pattern, network change), the average reconstruction error shifts too --
ADWIN detects that shift statistically and signals "this client should
retrain / rejoin the next federated round with fresh data" instead of
silently drifting into either missed detections or alert fatigue.
"""

from river.drift import ADWIN


class DriftMonitor:
    def __init__(self, delta=0.002):
        self.detector = ADWIN(delta=delta)
        self.drift_points = []
        self.n_seen = 0

    def update(self, error_value):
        """Feed one reconstruction-error value (float) at a time."""
        self.n_seen += 1
        in_drift = self.detector.update(error_value)
        if in_drift:
            self.drift_points.append(self.n_seen)
        return in_drift

    def has_drifted_recently(self, window=500):
        if not self.drift_points:
            return False
        return (self.n_seen - self.drift_points[-1]) < window

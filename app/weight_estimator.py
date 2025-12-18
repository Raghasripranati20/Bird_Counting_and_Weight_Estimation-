import numpy as np

class WeightEstimator:
    def __init__(self):
        self.history = {}

    def compute_weight_index(self, track_id, bbox):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        if track_id not in self.history:
            self.history[track_id] = []

        self.history[track_id].append(area)

        smoothed = np.mean(self.history[track_id][-10:])
        confidence = min(1.0, len(self.history[track_id]) / 10)

        return round(smoothed / 1000, 3), round(confidence, 2)

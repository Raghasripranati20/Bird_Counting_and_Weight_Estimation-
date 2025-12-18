import numpy as np
from filterpy.kalman import KalmanFilter

class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=4)
        self.kf.x = np.array(bbox)
        self.kf.F = np.eye(4)
        self.kf.H = np.eye(4)
        self.kf.P *= 100
        self.age = 0
        self.hits = 1
        self.bbox = bbox

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.bbox = self.kf.x.astype(int).tolist()
        return self.bbox

    def update(self, bbox):
        self.kf.update(np.array(bbox))
        self.hits += 1
        self.bbox = bbox


class MultiObjectTracker:
    def __init__(self, iou_thresh=0.3):
        self.tracks = []
        self.next_id = 0
        self.iou_thresh = iou_thresh

    def iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    def update(self, detections):
        updated_tracks = []

        for track in self.tracks:
            track.predict()

        for det in detections:
            best_iou, best_track = 0, None
            for track in self.tracks:
                iou_val = self.iou(track.bbox, det["bbox"])
                if iou_val > best_iou:
                    best_iou, best_track = iou_val, track

            if best_iou > self.iou_thresh:
                best_track.update(det["bbox"])
                updated_tracks.append(best_track)
            else:
                new_track = Track(self.next_id, det["bbox"])
                self.next_id += 1
                updated_tracks.append(new_track)

        self.tracks = updated_tracks
        return self.tracks

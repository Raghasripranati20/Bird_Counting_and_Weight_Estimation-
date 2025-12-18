from fastapi import FastAPI, UploadFile, File
import cv2, os, json
from app.detector import BirdDetector
from app.tracker import MultiObjectTracker
from app.weight_estimator import WeightEstimator
from app.video_utils import draw_annotations

app = FastAPI()

detector = BirdDetector()
tracker = MultiObjectTracker()
weight_estimator = WeightEstimator()

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/analyze_video")
def analyze_video(video: UploadFile = File(...)):
    os.makedirs("outputs", exist_ok=True)
    video_path = f"outputs/{video.filename}"

    with open(video_path, "wb") as f:
        f.write(video.file.read())

    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(
        "outputs/annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (int(cap.get(3)), int(cap.get(4)))
    )

    time_series = []
    weights = {}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 5 == 0:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            for t in tracks:
                w, conf = weight_estimator.compute_weight_index(t.id, t.bbox)
                weights[t.id] = {"weight_index": w, "confidence": conf}

            time_series.append({"time": frame_idx//5, "count": len(tracks)})
            frame = draw_annotations(frame, tracks, len(tracks))
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    response = {
        "counts": time_series,
        "tracks_sample": [
            {"track_id": k, "bbox": v["weight_index"]}
            for k,v in list(weights.items())[:3]
        ],
        "weight_estimates": {
            "unit": "index",
            "per_bird": weights
        },
        "artifacts": {
            "annotated_video": "outputs/annotated_video.mp4"
        }
    }

    with open("outputs/sample_response.json", "w") as f:
        json.dump(response, f, indent=2)

    return response

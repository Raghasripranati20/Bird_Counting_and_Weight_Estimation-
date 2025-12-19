# Bird_Counting_and_Weight_Estimation-

## Overview

This project implements a **bird counting and weight estimation pipeline** using computer vision and deep learning. It processes poultry videos to produce:

1. **Annotated video** with bounding boxes, tracking IDs, and count overlay  
2. **JSON response** with bird counts, sample tracks, and estimated relative weight indices  

The pipeline uses YOLOv8 for detection, multi-object tracking, and a custom weight estimator.

> ⚠️ Note: Due to time/environment constraints, the outputs in this repository are **sample placeholders**. Actual annotated video and JSON are generated when running the pipeline locally.

---

## Folder Structure

Bird_Counting_and_Weight_Estimation/
├── app/ # Source code
│ ├── main.py
│ ├── detector.py
│ ├── tracker.py
│ ├── weight_estimator.py
│ └── video_utils.py
├── datasets/
│ └── poultry_cctv/
│ ├── poultry_cctv.mp4
│ └── dataset_info.txt
├── outputs/
│ ├── annotated_video_placeholder.txt
│ └── sample_response.json
├── requirements.txt
└── README.md



## Installation

1. Clone the repository:

```bash
git clone https://github.com/Raghasripranati20/Bird_Counting_and_Weight_Estimation.git
cd Bird_Counting_and_Weight_Estimation

2. Create virtual environment (optional):
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac

3.Install dependencies:
pip install -r requirements.txt

Running the Project
Using FastAPI Server
1.Start server:
uvicorn app.main:app --reload
2.Open Swagger UI:
http://127.0.0.1:8000/docs
3.Use /analyze_video POST endpoint to upload a video (e.g., datasets/poultry_cctv/poultry_cctv.mp4).
4.Outputs will be saved in outputs/ folder.

Offline Pipeline (Optional)
Run:
python run_pipeline.py
Generates the same outputs without starting a server.

Outputs:

Annotated video: outputs/annotated_video.mp4 (placeholder in repo)
JSON response: outputs/sample_response.json (sample placeholder)
Real outputs are generated at runtime when the pipeline executes.

Notes
YOLOv8 weights are auto-downloaded on first run.

Large video files are excluded to keep the repo lightweight.

Python >= 3.9 recommended. OpenCV required.

Author
Raghasripranati Daggubati – Internship  Assignment submission
Contact: raghasri20@gmail.com


  











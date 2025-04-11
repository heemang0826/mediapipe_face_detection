# MediaPipe Face Detection on WIDER FACE Dataset

This project utilizes [Google MediaPipe's](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) published **face detection model** to run inference via its official Python API. The model is evaluated on the **WIDER FACE** validation dataset.

---

## 📁 Project Structure

```
project_root/
├── mediapipe_face_detection/
│   ├── detector.tflite              # Pretrained MediaPipe face detection model
│   ├── main.py                      # Runs inference and saves results
│   ├── eval.py                      # Evaluates results using AP / precision / recall
│   ├── tool.py                      # Utility for saving annotations
│   ├── visualize.py                 # Draws bounding boxes
│   ├── requirements.txt             # Minimal dependencies
│   ├── LICENSE                      # License info (Apache 2.0)
│   ├── .gitignore
│   └── outputs/                     # Inference outputs
│       ├── images/                  # Inference result images
│       └── labels/                  # Predicted bounding boxes in text format
├── venv_mediapipe/                  # Your Python virtual environment
└── wider_face/
    ├── wider_face_split/           # Contains annotation files (e.g. wider_face_val_bbx_gt.txt)
    ├── WIDER_test/                 # Test images
    ├── WIDER_train/                # Training images
    └── WIDER_val/                  # Validation images
```

---

## 📦 Setup Instructions

### 1. Clone this Repository

```bash
git clone https://github.com/heemang0826/mediapipe_face_detection.git
```

### 2. Install WIDER FACE Dataset

Download the [WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/) and place it as follows:

```
project_root/wider_face/
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv_mediapipe
source venv_mediapipe/bin/activate
```

### 4. Install Dependencies

```bash
python3 -m pip install -r mediapipe_face_detection/requirements.txt
```

---

## 🚀 Running Inference

```bash
python3 mediapipe_face_detection/main.py
```

This will:

- Load the MediaPipe face detector model `detector.tflite`
- Perform inference on WIDER FACE images
- Save result images and detection annotations to `outputs/`

---

## 📊 Evaluation

```bash
python3 mediapipe_face_detection/eval.py
```

This script compares your predictions against ground truth labels and computes:

- Average Precision (AP)
- Precision
- Recall

---

## 📎 Notes

- This code uses MediaPipe's **official face detection TFLite model**.
- License: [Apache 2.0](https://github.com/google/mediapipe/blob/master/LICENSE)
- Predictions are saved in WIDER FACE-like format: `x, y, w, h, score`

---

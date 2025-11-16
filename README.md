# Face Detection & Recognition Project

A Python-based face detection and real-time recognition system using OpenCV, deep learning embeddings, and SVM classification.

## Overview

This project detects faces in images and video streams, extracts 128-dimensional embeddings using OpenFace, and classifies them against a trained dataset using Support Vector Machine (SVM).

**Key features:**
- Automatic face detection in images and live video
- Face embedding extraction and storage
- SVM-based face classification
- Real-time webcam preview with live recognition
- One-click batch launchers for easy execution
- Support for multiple camera sources and output formats

## Project Structure

```
face-dedction/
├── dataset/               # Training data (labeled subdirectories: aakash/, bagaria/, bhatia/, modi/, rathi/, unknown/)
├── face_detection_model/  # Pre-trained face detection model files
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── images/                # Sample test images
│   ├── openface.jpg
│   └── output1.png
├── output/                # Generated outputs (embeddings, models, results)
│   ├── embeddings.pickle  # Face embeddings and names
│   ├── recognizer         # Trained SVM model
│   ├── le.pickle          # Label encoder (class names)
│   ├── result.jpg         # Sample annotated image
│   └── ...
├── openface_nn4.small2.v1.t7  # Pre-trained face embedding model
├── extract_embeddings.py  # Step 1: extract face embeddings from dataset
├── train_model.py         # Step 2: train SVM classifier
├── recognize_image.py     # Step 3a: recognize faces in a static image
├── recognize_video.py     # Step 3b: recognize faces in live video
├── run_live.bat           # One-click launcher: open webcam
├── run_pipeline.bat       # One-click launcher: extract → train → demo
├── create_shortcuts.ps1   # PowerShell script to create Desktop shortcuts
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Quick Start (One-Click)

### Windows (Easiest)

1. **Live Webcam Demo:**
   - Double-click `run_live.bat` to start the webcam and see real-time face recognition
   - Press `q` in the preview window to quit

2. **Full Pipeline Demo:**
   - Double-click `run_pipeline.bat` to extract embeddings, train the model, and run recognition on a sample image
   - Results saved to `output/result.jpg`

3. **Create Desktop Shortcuts** (optional):
   - Double-click `create_shortcuts.ps1` (or right-click → Run with PowerShell)
   - Creates "Run Live" and "Run Pipeline" shortcuts on your Desktop for even quicker access

## Setup (Manual)

### Prerequisites

- Python 3.7+ (tested on Python 3.13)
- Windows 10+ (for batch/PowerShell launchers)

### Installation

1. **Clone or download** this project to your machine

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Usage

### Full Pipeline (Train & Recognize)

Run these commands in order:

#### 1. Extract Face Embeddings
Analyzes all images in `dataset/` and extracts facial embeddings.
```powershell
python extract_embeddings.py
```
**Output:** `output/embeddings.pickle` (face vectors + names)

#### 2. Train Classifier
Trains an SVM model on the extracted embeddings.
```powershell
python train_model.py
```
**Output:** `output/recognizer` (trained model), `output/le.pickle` (label encoder)

#### 3a. Recognize in a Static Image
Detects and recognizes faces in an image file.
```powershell
# Show result in a window
python recognize_image.py --image images\openface.jpg

# Save annotated result to file
python recognize_image.py --image images\openface.jpg --output output\result.jpg
start output\result.jpg
```

#### 3b. Recognize in Live Video
Opens your webcam and shows live face recognition.
```powershell
# Default camera (camera 0)
python recognize_video.py

# Alternative camera
python recognize_video.py --src 1

# Save one annotated snapshot
python recognize_video.py --snapshot output\snapshot.jpg

# Save full annotated video (headless, no preview)
python recognize_video.py --no-display --output output\annotated.avi

# Adjust detection confidence threshold
python recognize_video.py --confidence 0.3
```

**In the video window, press `q` to quit.**

### CLI Options for `recognize_video.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--src` | `0` | Camera index (0, 1, 2, ...) or path to video file |
| `--no-display` | - | Run headless (no preview window) |
| `--output` | - | Save annotated video to file |
| `--snapshot` | - | Save one annotated frame and exit |
| `--confidence` | `0.5` | Min detection confidence (0–1, lower = more detections) |

## Dataset Structure

Place training images in subdirectories named by person:

```
dataset/
├── aakash/          # All Aakash's images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── bagaria/
│   ├── image1.jpg
│   └── ...
├── bhatia/
├── modi/
├── rathi/
└── unknown/         # Non-face or unknown images
```

Re-run `extract_embeddings.py` and `train_model.py` after adding new images.

## Troubleshooting

### Camera Issues

**Problem:** "can't grab frame" or `AttributeError: 'NoneType' object has no attribute 'shape'`

**Solutions:**
- Close apps using the camera (Teams, Zoom, browser tabs)
- Check Windows privacy: Settings → Privacy → Camera → allow desktop apps
- Try a different camera: `python recognize_video.py --src 1`
- Test camera access:
  ```powershell
  python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print('ret:', ret); cap.release()"
  ```

### Old Pickle Incompatibility

**Problem:** `ModuleNotFoundError: No module named 'sklearn.svm.classes'`

**Solution:** Remove old pickle files and retrain:
```powershell
del output\recognizer.pickle
python train_model.py
```

### No Faces Detected

**Problem:** Faces aren't being recognized or detected

**Checks:**
- Ensure `dataset/` contains labeled subfolders with clear face images (at least 5–10 per person)
- Check that `openface_nn4.small2.v1.t7` and `face_detection_model/*` files exist and are not corrupted
- Lower the confidence threshold: `python recognize_video.py --confidence 0.3`
- Verify training completed successfully: look for `output/recognizer` and `output/le.pickle`

### Virtualenv Issues

**Problem:** Scripts can't find Python after activating virtualenv

**Solution:**
- Ensure virtualenv is activated: `.venv\Scripts\Activate.ps1`
- Use absolute path to virtualenv Python: `C:\path\to\.venv\Scripts\python.exe`
- Batch files (`run_*.bat`) automatically detect `.venv` if present

## Dependencies

See `requirements.txt`:
- **numpy** — numerical computing
- **opencv-python** — computer vision (face detection, image processing)
- **imutils** — OpenCV convenience functions
- **scikit-learn** — SVM classifier and label encoding

## Performance Tips

- **Faster inference:** Lower `--confidence` threshold to skip weak detections
- **Smaller models:** Use `opencv-contrib-python` with hardware acceleration (CUDA/OpenCL) if available
- **Batch processing:** For many images, modify `extract_embeddings.py` to use GPU (requires CUDA)
- **Video recording:** Use H.264 codec instead of XVID for better compression: modify `fourcc` in `recognize_video.py`

## Advanced Usage

### Custom Dataset Training

1. Create labeled image folders in `dataset/` (one folder per person)
2. Run `python extract_embeddings.py`
3. Run `python train_model.py`
4. Run `python recognize_video.py` to test live

### Adjusting Detection Sensitivity

Edit the confidence threshold in `recognize_image.py` or use `--confidence` flag in `recognize_video.py`:
- **0.3–0.5:** Lower threshold, more detections (may include false positives)
- **0.7–0.9:** Higher threshold, fewer detections (only strong faces)

### Using a Video File Instead of Webcam

```powershell
python recognize_video.py --src C:\path\to\video.mp4
```

## Architecture

1. **Face Detection:** SSD (Single Shot MultiBox Detector) on 300×300 frames
2. **Face Embedding:** OpenFace (128-dimensional vector) on 96×96 aligned faces
3. **Classification:** Linear SVM trained on embeddings
4. **Output:** Bounding box + confidence percentage for each detected face

## License

Check `LICENSE` for details.

## Author

**sabari2327** — GitHub repository: [face-dedction](https://github.com/sabari2327/face-dedction)

## References

- **OpenFace:** [cmusatyalab/openface](https://github.com/cmusatyalab/openface)
- **OpenCV:** [opencv/opencv](https://github.com/opencv/opencv)
- **scikit-learn:** [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)

---

## Quick Cheat Sheet

| Task | Command |
|------|---------|
| Install deps | `python -m pip install -r requirements.txt` |
| Extract embeddings | `python extract_embeddings.py` |
| Train model | `python train_model.py` |
| Recognize in image | `python recognize_image.py --image images\openface.jpg` |
| Open webcam | `python recognize_video.py` |
| Save image result | `python recognize_image.py --image images\openface.jpg --output output\result.jpg` |
| Save video result | `python recognize_video.py --no-display --output output\video.avi` |
| One-click webcam | Double-click `run_live.bat` |
| One-click full demo | Double-click `run_pipeline.bat` |
| Create Desktop shortcuts | `powershell -NoProfile -ExecutionPolicy Bypass -File create_shortcuts.ps1 -Force` |

---

**Last Updated:** November 2025



---
title: DermAI Skin Classifier
emoji: 🌍
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
---

# 🩺 DermAI — Skin Disease Classifier

An AI-powered web app that classifies 22 different skin conditions from a single image, built on an EfficientNetB5 backbone with test-time augmentation (TTA) for improved accuracy.

**🔗 Live app:** [Hugging Face Space](https://huggingface.co/spaces/Gupta-Tarun/dermai-skin-classifier)
**💻 Source:** [GitHub repo](https://github.com/Gupta-Tarun7/Skin_Disease_Classifier)

> ⚠️ **Not a medical diagnosis.** This tool is for educational and research purposes only. Always consult a qualified dermatologist for any skin concern.

---

## Pipeline
Kaggle dataset (22 classes)
│
▼
Augmentation (MixUp, rotation, zoom, brightness jitter)
│
▼
EfficientNetB5 — Phase 1: train classification head (backbone frozen)
│
▼
Phase 2: fine-tune (unfreeze last 150 layers, low LR)
│
▼
Evaluation — TTA (8-pass averaging), confusion matrix, per-class AUC, Grad-CAM
│
▼
Flask API — serves predictions with 3-pass live TTA
│
▼
Docker — containerized for reproducible deployment
│
▼
Hugging Face Spaces — live deployment

---

## Model metrics

| Metric | Value |
|---|---|
| Top-1 accuracy | 62.81% |
| Top-3 accuracy | 82.41% |
| AUC (one-vs-rest) | 0.9426 |
| Classes | 22 |
| Parameters | ~29.7M |
| Input resolution | 456×456 |
| Training images | 11,128 |
| Validation images | 2,770 |
| Test images | 1,546 |
| TTA passes (evaluation) | 8 |
| TTA passes (live inference) | 3 |

---

## Tech stack

- **Model:** TensorFlow / Keras, EfficientNetB5
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker, Hugging Face Spaces
- **Versioning:** Git, Git LFS (for the ~280MB model file)

---

## Acknowledgements

Built with the help of **Claude AI** — used for building the frontend UI and as a debugging partner while troubleshooting deployment issues (Git LFS resolution, Keras version compatibility, Linux case-sensitivity bugs).
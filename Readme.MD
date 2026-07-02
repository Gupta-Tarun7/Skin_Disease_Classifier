# DermAI — Skin Disease Classifier

A single Flask app that serves both the prediction API and a custom
HTML/CSS/JS frontend (no separate Streamlit process). On each prediction
it runs **3 augmented passes** (test-time augmentation) per image and
averages the results for a steadier confidence score.

## Project structure

```
skin_app/
├── app.py                          # Flask backend + frontend server
├── requirements.txt
├── skin_model_phase2_improved.keras  # place your model file here
├── templates/
│   └── index.html                  # page markup
└── static/
    ├── style.css                   # purple/black theme
    └── app.js                      # upload, drag-and-drop, results rendering
```

## 1. Setup

Place `skin_model_phase2_improved.keras` in the same folder as `app.py`
(or update `MODEL_PATH` in `app.py` if you keep it elsewhere).

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
python app.py
```

Wait for the log line `Model loaded. Input shape: ...` — the first load
takes ~10–40 seconds since EfficientNetB5 is a fairly large model and each
prediction now runs 3 forward passes instead of 1.

Then open **http://localhost:5000** in your browser. That's it — one
process, one port, frontend and API together.

## How it works

- **Upload** — drag an image onto the dashed dropzone, or click it to
  browse. The preview appears immediately while the request is in flight,
  with an animated scan line over it.
- **Prediction** — `app.js` POSTs the file to `/predict`. The Flask backend:
  1. Resizes the image to 456×456.
  2. Generates 3 independently-augmented copies (rotation ±20°, horizontal
     flip, ±10% zoom, brightness jitter — matching the training notebook's
     TTA recipe).
  3. Runs each copy through the model and averages the 3 resulting
     probability vectors.
  4. Returns the top 3 classes from the averaged result, plus each class's
     individual confidence on each of the 3 passes.
- **Display** — the UI shows the 3 individual pass confidences for the top
  prediction converging into the averaged bar, then lists the top-3 results
  as ranked cards.
- **Dashboard** — model name, parameter count, and evaluation metrics are
  fetched from `/model-info` and shown in the header badges and metrics
  strip, so they always reflect the model actually being served.

## Model metrics shown on the dashboard

Pulled from the notebook's evaluation on the held-out test set (1,546
images). The dashboard shows the notebook's **8-pass TTA evaluation**
numbers as the benchmark for "this model with TTA enabled":

| Metric | Value |
|---|---|
| Top-1 Accuracy (8-pass TTA) | 60.67% |
| Top-3 Accuracy (8-pass TTA) | 81.24% |
| AUC (One-vs-Rest) | 0.9431 |
| Classes | 22 |
| Training images | 11,128 |
| Validation images | 2,770 |
| Test images | 1,546 |
| Parameters | ~29.7M |

**Note:** the notebook's official TTA evaluation averaged **8** passes per
image across the full test set. This app runs **3** passes per request — a
latency tradeoff for live single-image inference. The dashboard's headline
metrics are shown as the "TTA-enabled" benchmark for the model, not a
re-measurement of the live 3-pass config; per-image results from the live
app will vary a little run-to-run since each request draws a fresh random
sample of 3 augmentations.

These live in `MODEL_METRICS` in `app.py` — update them there if you
retrain, re-evaluate, or change `TTA_PASSES`.

## Tuning TTA passes vs. speed

`TTA_PASSES` near the top of `app.py` controls how many augmented passes
run per request. Each pass is one full forward pass through EfficientNetB5,
so request time scales roughly linearly with this number (3 passes ≈
2 seconds on a typical CPU once the model is warmed up; 5 passes ≈ 3.3s;
8 passes ≈ 5.3s). Raise it for steadier averaged predictions at the cost of
slower responses, or lower it for snappier single-image use.

## Notes

- Class names are hard-coded in `app.py` in the exact alphabetical order
  Keras' `flow_from_directory` used during training (`Acne`,
  `Actinic_Keratosis`, ... `Warts` — 22 classes total). If you retrain with
  a different folder structure, update `CLASS_NAMES` to match.
- The model was loaded with `compile=False` since it was trained with a
  custom focal-loss function — this is safe for inference-only use and
  avoids needing to redefine the custom loss just to load the weights.
- This is a **research/educational demo only** — it is not a medical
  device and should never be used for actual diagnosis.
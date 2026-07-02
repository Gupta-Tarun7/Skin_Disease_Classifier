"""
Flask backend + frontend server for the Skin Disease Classifier.

Serves the DermAI single-page UI (templates/index.html) and exposes the
prediction API on the same Flask app/port — no separate frontend process
needed. Loads the trained EfficientNetB5 Keras model once at startup and
performs test-time augmentation (TTA) on each prediction: the uploaded
image is augmented N times (random rotation, horizontal flip, zoom,
brightness jitter — same recipe as the training notebook), each augmented
version is run through the model, and the softmax outputs are averaged
before ranking the top-3 classes.
"""

import io
import os
import random

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, ImageEnhance

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------------------------------------------------------
# Config — must match training notebook exactly
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "skin_model_phase2_improved.keras")
IMG_SIZE = 456  # EfficientNetB5 input size used during training
TTA_PASSES = 3  # number of augmented forward passes averaged per prediction

# Class names in the exact order produced by Keras' flow_from_directory,
# which sorts class subfolder names alphabetically.
CLASS_NAMES = [
    "Acne",
    "Actinic_Keratosis",
    "Benign_tumors",
    "Bullous",
    "Candidiasis",
    "DrugEruption",
    "Eczema",
    "Infestations_Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrh_Keratoses",
    "SkinCancer",
    "Sun_Sunlight_Damage",
    "Tinea",
    "Unknown_Normal",
    "Vascular_Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts",
]

MODEL_NAME = "EfficientNetB5 — Skin Disease Classifier v2 (TTA×3)"

# NOTE on these numbers: the notebook's official TTA evaluation used 8
# augmented passes per image, averaged over the full 1,546-image test set.
# This app runs 3 passes per request (a latency/accuracy tradeoff for live
# inference). 3-pass accuracy on a single image will vary a bit run-to-run
# since it's a smaller, randomized sample of augmentations — these reported
# metrics are the notebook's full 8-pass evaluation and are shown as the
# benchmark for "this model with TTA," not a re-measurement of 3 passes.
MODEL_METRICS = {
    "top1_accuracy": 0.6067,
    "top3_accuracy": 0.8124,
    "auc_ovr": 0.9431,
    "num_classes": 22,
    "train_images": 11128,
    "val_images": 2770,
    "test_images": 1546,
    "params": 29710861,
    "input_size": IMG_SIZE,
    "tta_passes_eval": 8,
    "tta_passes_live": TTA_PASSES,
}

app = Flask(__name__)
CORS(app)  # kept for convenience if you ever call the API from a different origin

print("Loading model, this can take a little while...")
# compile=False: we only need the model for inference, so we skip
# reconstructing the custom focal-loss training objective entirely.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded. Input shape:", model.input_shape, "Output shape:", model.output_shape)

# "Warm up" the model with a dummy forward pass so the first real
# request from a user isn't slowed down by lazy graph building.
_ = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), verbose=0)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def load_base_image(file_bytes: bytes) -> Image.Image:
    """Decode raw image bytes and resize to the model's expected input size."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return img


def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply one random augmentation, matching the notebook's TTA recipe:
    rotation_range=20, horizontal_flip=True, zoom_range=0.1,
    brightness_range=[0.9, 1.1].
    """
    out = img

    # Random rotation within +/-20 degrees, reflect-padded like training
    angle = random.uniform(-20, 20)
    out = out.rotate(angle, resample=Image.BILINEAR, fillcolor=None, expand=False)

    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    # Random zoom +/-10%: crop or pad then resize back to IMG_SIZE
    zoom = random.uniform(0.9, 1.1)
    new_size = int(IMG_SIZE * zoom)
    out = out.resize((new_size, new_size), Image.BILINEAR)
    if new_size >= IMG_SIZE:
        left = (new_size - IMG_SIZE) // 2
        top = (new_size - IMG_SIZE) // 2
        out = out.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    else:
        canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        offset = ((IMG_SIZE - new_size) // 2, (IMG_SIZE - new_size) // 2)
        canvas.paste(out, offset)
        out = canvas

    # Random brightness jitter in [0.9, 1.1]
    brightness = random.uniform(0.9, 1.1)
    out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


def predict_with_tta(img: Image.Image, n_passes: int = TTA_PASSES):
    """
    Run n_passes augmented forward passes through the model and average
    the resulting softmax outputs. Returns (averaged_preds, per_pass_preds).
    """
    per_pass = []
    for _ in range(n_passes):
        aug = augment_image(img)
        arr = np.array(aug, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)  # EfficientNet-specific normalization
        preds = model.predict(arr, verbose=0)[0]
        per_pass.append(preds)

    per_pass = np.stack(per_pass, axis=0)  # shape (n_passes, 22)
    averaged = np.mean(per_pass, axis=0)
    return averaged, per_pass


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({"name": MODEL_NAME, "metrics": MODEL_METRICS, "classes": CLASS_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded. Send the image under the 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        file_bytes = file.read()
        base_img = load_base_image(file_bytes)
    except Exception as exc:
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    averaged, per_pass = predict_with_tta(base_img, TTA_PASSES)

    top3_idx = np.argsort(averaged)[::-1][:3]
    results = [
        {
            "rank": rank + 1,
            "class": CLASS_NAMES[idx],
            "confidence": float(averaged[idx]),
            # how each individual TTA pass scored this same class —
            # lets the UI show the averaging actually happening
            "pass_confidences": [float(p[idx]) for p in per_pass],
        }
        for rank, idx in enumerate(top3_idx)
    ]

    return jsonify({"predictions": results, "tta_passes": TTA_PASSES})


if __name__ == "__main__":
    # threaded=True so the warmed-up model can serve while staying responsive
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
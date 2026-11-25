#!/usr/bin/env python3
"""Run the EpiTrack TFLite ensemble on one photo, the way the Flutter app does.

Use it to check the app's on-device output on the same photo: compare the
app's "[RESULT]" log lines (adb logcat -s flutter) with what this prints.

Preprocessing matches mobile/lib/services/tflite_inference_service.dart:
  1. decode, and apply the EXIF orientation (upright, as the app does),
  2. resize to 380x380 with the Dart `image` package's bilinear algorithm,
     re-implemented below (point sampling, no antialiasing, no half-pixel
     offset, result truncated to 0-255 integers),
  3. RGB, NHWC, raw 0-255 float32. No /255, no mean/std.
Then every model in epitrack_config.json gets the same input, and their
15 probabilities are averaged equally.

Expect small differences from the app, not an exact match: PIL and the Dart
`image` package decode JPEGs with slightly different code, so pixels can be
off by a level or two.

--keras-resize also runs the training-time resize (Keras load_img /
flow_from_directory default: nearest-neighbour to 380x380) and prints both
results side by side.

Usage:
  ~/FYP/venv/bin/python ~/FYP/ml/tools/verify_tflite.py photo.jpg
  ~/FYP/venv/bin/python ~/FYP/ml/tools/verify_tflite.py photo.jpg --keras-resize
  ~/FYP/venv/bin/python ~/FYP/ml/tools/verify_tflite.py photo.jpg --models-dir other/dir
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Hide TF's start-up noise (CUDA/cuDNN "errors" on a machine without a GPU).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = 380
NUM_CLASSES = 15
DEFAULT_MODELS_DIR = Path.home() / "FYP" / "mobile" / "assets" / "models"
EXIF_ORIENTATION_TAG = 0x0112


# --------------------------------------------------------------------------
# Config and labels (same checks as the app)
# --------------------------------------------------------------------------

def load_config(models_dir):
    config_path = models_dir / "epitrack_config.json"
    labels_path = models_dir / "class_indices_15.json"
    for path in (config_path, labels_path):
        if not path.is_file():
            sys.exit(f"error: {path} not found")

    config = json.loads(config_path.read_text())
    class_indices = json.loads(labels_path.read_text())

    if len(class_indices) != NUM_CLASSES:
        sys.exit(f"error: {labels_path.name} has {len(class_indices)} "
                 f"classes, expected {NUM_CLASSES}")
    labels = [None] * NUM_CLASSES
    for name, index in class_indices.items():
        if not isinstance(index, int) or not 0 <= index < NUM_CLASSES \
                or labels[index] is not None:
            sys.exit(f"error: {labels_path.name}: bad or duplicate index "
                     f"{index!r} for {name!r}")
        labels[index] = name

    known = set(labels)
    named = list(config["malignant_classes"])
    for names in config["category_groups"].values():
        named += names
    unknown = [n for n in named if n not in known]
    if unknown:
        sys.exit(f"error: epitrack_config.json names classes not in "
                 f"{labels_path.name}: {unknown}")

    missing = [m for m in config["models"] if not (models_dir / m).is_file()]
    if missing:
        sys.exit(f"error: models listed in epitrack_config.json but missing "
                 f"from {models_dir}: {missing}")
    return config, labels


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def open_upright_rgb(path):
    """Decode, apply EXIF orientation, convert to 8-bit RGB.

    Returns (image, exif_orientation) so callers can report the orientation.
    """
    image = Image.open(path)
    orientation = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB"), orientation


def dart_bilinear_resize(rgb, size=INPUT_SIZE):
    """Dart `image` 4.x copyResize(..., interpolation: linear), exactly.

    For each output pixel x: fx = x * (srcWidth / size), ix = floor(fx),
    kx = fx - ix, nx = min(ix + 1, srcWidth - 1) (same for y), then
        v = icc + kx*(inc - icc + ky*(icc + inn - icn - inc)) + ky*(icn - icc)
    truncated to an integer, where icc=(ix,iy), inc=(nx,iy), icn=(ix,ny),
    inn=(nx,ny). Plain point sampling: no antialiasing when shrinking.
    """
    src = np.asarray(rgb, dtype=np.float64)  # (H, W, 3)
    src_h, src_w = src.shape[:2]

    fx = np.arange(size, dtype=np.float64) * (src_w / size)
    ix = fx.astype(np.int64)
    kx = (fx - ix)[None, :, None]
    nx = np.minimum(ix + 1, src_w - 1)

    fy = np.arange(size, dtype=np.float64) * (src_h / size)
    iy = fy.astype(np.int64)
    ky = (fy - iy)[:, None, None]
    ny = np.minimum(iy + 1, src_h - 1)

    icc = src[iy][:, ix]
    inc = src[iy][:, nx]
    icn = src[ny][:, ix]
    inn = src[ny][:, nx]
    value = icc + kx * (inc - icc + ky * (icc + inn - icn - inc)) \
        + ky * (icn - icc)
    return np.trunc(value).astype(np.uint8)


def keras_nearest_resize(rgb, size=INPUT_SIZE):
    """What training saw: Keras load_img(target_size=...) with its default
    interpolation="nearest", i.e. PIL resize with NEAREST."""
    return np.asarray(rgb.resize((size, size), Image.NEAREST), dtype=np.uint8)


def to_model_input(pixels):
    """(380, 380, 3) uint8 -> (1, 380, 380, 3) float32, raw 0-255."""
    return pixels.astype(np.float32)[None, ...]


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------

def load_interpreters(models_dir, model_files, threads):
    import tensorflow as tf  # slow import: only once arguments are valid

    interpreters = []
    for name in model_files:
        interpreter = tf.lite.Interpreter(
            model_path=str(models_dir / name), num_threads=threads)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        if list(inp["shape"]) != [1, INPUT_SIZE, INPUT_SIZE, 3] \
                or list(out["shape"]) != [1, NUM_CLASSES]:
            sys.exit(f"error: {name} has input {list(inp['shape'])} and "
                     f"output {list(out['shape'])}, expected "
                     f"[1, {INPUT_SIZE}, {INPUT_SIZE}, 3] and "
                     f"[1, {NUM_CLASSES}]")
        interpreters.append((name, interpreter, inp["index"], out["index"]))
    return interpreters


def run_ensemble(interpreters, model_input, labels, config):
    per_model = []
    for name, interpreter, in_index, out_index in interpreters:
        start = time.perf_counter()
        interpreter.set_tensor(in_index, model_input)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_index)[0].astype(np.float64)
        per_model.append((name, probs, (time.perf_counter() - start) * 1000))

    averaged = np.mean([probs for _, probs, _ in per_model], axis=0)
    by_label = dict(zip(labels, averaged))
    malignancy = sum(by_label[n] for n in config["malignant_classes"])
    category_scores = {
        category: sum(by_label[n] for n in names)
        for category, names in config["category_groups"].items()
    }
    # max() keeps the first on a tie, like the app's topCategory().
    category = max(category_scores, key=category_scores.get)
    return {
        "per_model": per_model,
        "averaged": averaged,
        "malignancy": malignancy,
        "malignant": malignancy >= config["malignancy_threshold"],
        "category_scores": category_scores,
        "category": category,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def pct(p):
    return f"{p * 100:6.2f}%"


def top_indices(probs, n):
    return list(np.argsort(-probs, kind="stable")[:n])


def print_single(result, labels, config, show_all):
    print("\nPer model (top-1):")
    for name, probs, ms in result["per_model"]:
        k = int(np.argmax(probs))
        print(f"  {name:28s} {labels[k]}  {pct(probs[k])}   ({ms:.0f} ms)")

    n = NUM_CLASSES if show_all else 3
    print(f"\nAveraged ensemble, top {n}:")
    for k in top_indices(result["averaged"], n):
        print(f"  {pct(result['averaged'][k])}  {labels[k]}")

    print(f"\nMalignancy sum: {result['malignancy']:.4f} "
          f"(threshold {config['malignancy_threshold']}) -> "
          f"{'FLAGGED' if result['malignant'] else 'not flagged'}")
    print(f"Category: {result['category']}")
    for category, score in result["category_scores"].items():
        print(f"  {pct(score)}  {category}")


def print_side_by_side(app, keras, labels, config, show_all, pixel_diff):
    print(f"\nInput pixels, app bilinear vs Keras nearest: mean |diff| "
          f"{pixel_diff.mean():.2f}, max {pixel_diff.max()} (0-255 levels)")

    print("\nPer model (top-1):        app (bilinear)                "
          "  keras (nearest)")
    for (name, a_probs, _), (_, k_probs, _) in zip(app["per_model"],
                                                   keras["per_model"]):
        a, k = int(np.argmax(a_probs)), int(np.argmax(k_probs))
        print(f"  {name:24s} {labels[a][:22]:22s} {pct(a_probs[a])}   "
              f"{labels[k][:22]:22s} {pct(k_probs[k])}")

    if show_all:
        rows = top_indices(app["averaged"], NUM_CLASSES)
    else:
        # The union of both top-3s, so a class that only one of them ranks
        # highly still shows up.
        rows = top_indices(app["averaged"], 3)
        rows += [k for k in top_indices(keras["averaged"], 3) if k not in rows]
    print(f"\n{'Averaged ensemble':40s} {'app':>8s} {'keras':>8s} "
          f"{'diff':>8s}")
    for k in rows:
        a, b = app["averaged"][k], keras["averaged"][k]
        print(f"  {labels[k][:38]:38s} {pct(a)} {pct(b)} "
              f"{(b - a) * 100:+7.2f}")

    threshold = config["malignancy_threshold"]
    print(f"\n{'Malignancy sum (threshold ' + str(threshold) + ')':40s} "
          f"{app['malignancy']:8.4f} {keras['malignancy']:8.4f} "
          f"{keras['malignancy'] - app['malignancy']:+8.4f}")
    print(f"{'Flagged':40s} {str(app['malignant']):>8s} "
          f"{str(keras['malignant']):>8s}")
    print(f"{'Category':40s} {app['category']} | {keras['category']}")

    a_top = labels[int(np.argmax(app["averaged"]))]
    k_top = labels[int(np.argmax(keras["averaged"]))]
    same = (a_top == k_top and app["category"] == keras["category"]
            and app["malignant"] == keras["malignant"])
    print("\nVerdict: " + ("same top class, category and flag."
                           if same else
                           "the resize method CHANGES the outcome."))


def main():
    parser = argparse.ArgumentParser(
        description="Run the EpiTrack TFLite ensemble on one photo, with "
                    "the app's preprocessing.")
    parser.add_argument("image", type=Path, help="JPEG (or other) photo")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="folder with epitrack_config.json, "
                             "class_indices_15.json and the .tflite files "
                             f"(default: {DEFAULT_MODELS_DIR})")
    parser.add_argument("--keras-resize", action="store_true",
                        help="also run the training-time resize (Keras "
                             "nearest) and compare side by side")
    parser.add_argument("--threads", type=int, default=2,
                        help="TFLite CPU threads per model (default: 2)")
    parser.add_argument("--all", action="store_true",
                        help="show all 15 classes, not just the top 3")
    args = parser.parse_args()

    if not args.image.is_file():
        sys.exit(f"error: {args.image} not found")
    models_dir = args.models_dir.expanduser()
    config, labels = load_config(models_dir)

    rgb, orientation = open_upright_rgb(args.image)
    print(f"Image: {args.image} ({rgb.width}x{rgb.height} after EXIF "
          f"orientation {orientation})")
    print(f"Models: {', '.join(config['models'])} "
          f"[{config.get('model_version', '?')}], threads={args.threads}")
    if args.keras_resize and orientation != 1:
        print("Note: both resizes use the upright photo, so only the resize "
              "differs. Keras load_img itself ignores EXIF orientation.")

    interpreters = load_interpreters(models_dir, config["models"],
                                     args.threads)

    app_pixels = dart_bilinear_resize(rgb)
    app = run_ensemble(interpreters, to_model_input(app_pixels), labels,
                       config)

    if not args.keras_resize:
        print_single(app, labels, config, args.all)
        return

    keras_pixels = keras_nearest_resize(rgb)
    keras = run_ensemble(interpreters, to_model_input(keras_pixels), labels,
                         config)
    pixel_diff = np.abs(app_pixels.astype(np.int16)
                        - keras_pixels.astype(np.int16))
    print_side_by_side(app, keras, labels, config, args.all, pixel_diff)


if __name__ == "__main__":
    main()

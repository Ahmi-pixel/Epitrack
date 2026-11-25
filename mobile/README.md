# EpiTrack (mobile)

Flutter app for on-device skin-lesion screening and tracking. It photographs a
lesion, classifies it with a 3-model TFLite ensemble (EfficientNet-B4,
DenseNet, ConvNeXt; 15 classes), flags possible malignancy, and keeps a
per-lesion scan history (SQLite) for comparing scans over time.

## Setup

1. Install Flutter 3.44 or newer (`flutter --version`).
2. Download the 3 models. They are **not in git**. Get them from Google Drive
   `EpiTrack_Data/models/15class-v1/` and put them in `mobile/assets/models/`:
   - `b4_15_fp16.tflite`
   - `densenet_15_fp16.tflite`
   - `convnext_15_fp16.tflite`

   `class_indices_15.json` and `epitrack_config.json` are already in the repo.
3. `cd mobile && flutter pub get && flutter test`
4. Connect an Android phone with USB debugging on, using a data-capable USB
   cable (charge-only cables don't work), then `flutter run`.

The app is a screening aid, not a diagnosis.

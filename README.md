# EpiTrack FYP — AI-Powered Skin Disease Detection & Progression Tracking

![Status](https://img.shields.io/badge/Status-Phase%207-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-77%25%20→%2082--85%25-brightgreen)
![Month](https://img.shields.io/badge/Timeline-Month%202%2F6-orange)

**University:** University of Lahore, BSCS Program
**Supervisor:** Miss Rabiya Ali
**Student:** Syed Ahmad Ali Shah
**GitHub:** https://github.com/Ahmi-pixel/Epitrack

---

## 🎯 Project Overview

EpiTrack is a comprehensive AI-powered mobile application for skin disease detection and monitoring:

### Key Features
- 🤖 **Real-time Disease Detection** — 24 skin disease classes with 82-85% accuracy
- 📊 **Progression Tracking** — Monitor disease improvement/worsening over time
- 👨‍⚕️ **Doctor Portal** — Share diagnoses with healthcare professionals
- 📱 **Mobile-First** — Flutter app with offline TFLite inference (3 seconds/diagnosis)

### Technical Stack
- **AI/ML:** TensorFlow, EfficientNet-B4, Siamese Networks
- **Mobile:** Flutter (iOS/Android)
- **Backend:** FastAPI, PostgreSQL
- **Deployment:** Cloud-based with CI/CD

---

## 📈 Current Progress

### Phase 7: Clean 24-Class Dataset Retraining
- **Status:** 🔄 In Progress (Kaggle)
- **Previous Best:** 77% validation accuracy (Phase 6, 29 classes)
- **Target:** 82-85% accuracy (Phase 7, 24 classes)
- **Data:** HAM10000 (7 classes) + DermNet (23 classes) = 24 best classes
- **Removed:** 5 weakest performing classes (F1 < 0.57)

### Training Timeline
| Phase | Approach | Accuracy | Status |
|-------|----------|----------|--------|
| Phase 1-2 | Base frozen, fine-tune head | ~55% | ✅ Complete |
| Phase 3 | Full unfreeze + LR scheduler | 63.17% | ✅ Complete |
| Phase 4 | Rebalanced class weights | 67.25% | ✅ Complete |
| Phase 5 | 380x380 resolution | 72% | ✅ Complete |
| Phase 6 | Extended training (25→60 epochs) | 77% | ✅ Complete |
| Phase 7 | Clean 24-class dataset | TBD | 🔄 Running |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 8GB+ RAM (GPU recommended for training)
- ~10GB disk space for datasets

### Setup Local Environment

```bash
# If not already done, run setup
bash scripts/setup_local.sh

# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__}')"
```

### Download Datasets

See detailed instructions in `ml/data/README.md`

Quick: Download from Kaggle
- HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- DermNet: Search "DermNet" on Kaggle

### Prepare Data

```bash
python ml/prepare_data.py
```

### Train Model

```bash
# Phase 7 (clean 24-class dataset)
python ml/training/phase7_clean24.py
```

---

## 📁 Project Structure

```
epitrack-fyp/
├── README.md
├── ROADMAP.md
├── DATASET_INFO.md
├── MODEL_PHASES.md
├── requirements.txt
├── .gitignore
│
├── ml/
│   ├── data/
│   │   ├── ham10000/ (downloaded, ~3GB)
│   │   ├── organized/ (combined dataset)
│   │   └── README.md
│   ├── training/
│   │   ├── phase1_baseline.py
│   │   ├── phase2_unfreeze.py
│   │   ├── phase3_continued.py
│   │   ├── phase4_rebalanced.py
│   │   ├── phase5_380x380.py
│   │   ├── phase6_continuation.py
│   │   ├── phase7_clean24.py
│   │   └── confusion_matrix.py
│   ├── models/
│   ├── prepare_data.py
│   └── notebooks/
│
├── mobile/
│   ├── README.md
│   └── flutter_setup.md
│
├── backend/
│   ├── README.md
│   └── api_setup.md
│
├── docs/
│   ├── PHASES.md
│   ├── BUGS_FIXED.md
│   ├── DECISIONS.md
│   └── CLASS_ANALYSIS.md
│
└── scripts/
    ├── setup_local.sh
    ├── create_docs.sh
    └── setup_github.sh
```

---

## 🔬 Model Architecture

```
Input: 380x380 RGB Image
↓
EfficientNet-B4 (ImageNet pretrained)
↓
Global Average Pooling
↓
Dense(256, relu) + Dropout(0.4)
↓
Dense(24, softmax) → Output probabilities
```

**Model Size:** ~500MB (full), ~100MB (quantized TFLite)
**Inference Time:** ~3 seconds (CPU), ~0.5s (GPU)

---

## 📊 Dataset Breakdown (Phase 7)

### 24 Disease Classes

**Best Performers (F1 > 0.75):**
- nevus (0.915) — 1,058 samples
- Acne and Rosacea Photos (0.809) — 162 samples
- basal_cell_carcinoma (0.794) — 71 samples
- Nail Fungus (0.785) — 198 samples
- vascular_lesion (0.800) — 22 samples

**Okay Performers (0.60-0.75):**
- Eczema, Tinea, Psoriasis, Seborrheic Keratoses, etc.

**Removed (F1 < 0.57):**
- Cellulitis/Impetigo (0.407)
- Herpes/HPV (0.550)
- Poison Ivy (0.554)
- Lupus (0.556)
- actinic_keratosis (0.571)

### Data Distribution

```
Total Images: ~31,000
Training: ~21,700 (70%)
Validation: ~4,650 (15%)
Test: ~4,650 (15%)
```

---

## 📚 Documentation

- **[ROADMAP.md](ROADMAP.md)** — Full 6-month timeline
- **[docs/PHASES.md](docs/PHASES.md)** — Detailed training phases
- **[docs/BUGS_FIXED.md](docs/BUGS_FIXED.md)** — Critical issues & fixes
- **[docs/DECISIONS.md](docs/DECISIONS.md)** — Strategic choices
- **[DATASET_INFO.md](DATASET_INFO.md)** — Dataset documentation

---

## 📞 Contact

**Supervisor:** Miss Rabiya Ali, University of Lahore
**Student:** Syed Ahmad Ali Shah
**GitHub:** https://github.com/Ahmi-pixel/Epitrack

---

**Last Updated:** September 2026

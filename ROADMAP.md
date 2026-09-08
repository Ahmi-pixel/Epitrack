# EpiTrack FYP — Complete 6-Month Roadmap

## 📊 Current Status
- **Month:** 2 of 6 (Training AI Model)
- **Phase:** 7 (Clean 24-class Dataset)
- **Current Accuracy:** 77% (Phase 6) → **Target: 82-85%**

---

## ✅ MONTH 1: SETUP + DATA PREPARATION

### Completed
- [x] Python 3.11 environment setup
- [x] Downloaded HAM10000 (10,015 images, 7 diseases)
- [x] Downloaded DermNet (20,000+ images, 23 diseases)
- [x] Combined datasets: 30 classes, ~35,000 images
- [x] Organized into train/val/test (70/15/15 split)
- [x] GitHub repository initialized

---

## 🔄 MONTH 2: TRAIN AI MODEL (To 75%+ Accuracy)

### Phase 1-6: Completed ✅
| Phase | Strategy | Accuracy |
|-------|----------|----------|
| 1-2 | Base frozen + unfreeze layers | ~55% |
| 3 | Full unfreeze + LR scheduler | 63.17% |
| 4 | Rebalanced class weights | 67.25% |
| 5 | 380x380 resolution | 72% |
| 6 | Extended training | 77% |

### Phase 7: Current 🔄
- Remove 5 weakest classes (F1 < 0.57)
- Retrain on 24 best classes
- Fresh from ImageNet weights
- Expected: 82-85% accuracy

### Bugs Fixed
1. **Double Rescaling** → Collapsed model (54.7%) — FIXED
2. **Aggressive Class Weights** → Melanoma bias — FIXED
3. **Kaggle Save Issues** → Lost checkpoints — FIXED

---

## 📱 MONTH 3: MOBILE APP

- [ ] Convert model to TFLite
- [ ] Flutter app setup
- [ ] Camera capture + diagnosis
- [ ] Grad-CAM visualization

---

## 📊 MONTH 4: PROGRESSION TRACKING

- [ ] Siamese network training
- [ ] Image similarity matching
- [ ] Timeline UI

---

## 🔧 MONTH 5: BACKEND + PORTAL

- [ ] FastAPI server
- [ ] User authentication
- [ ] Doctor portal

---

## 🚀 MONTH 6: DEPLOYMENT

- [ ] Testing & optimization
- [ ] Security hardening
- [ ] Production deployment

---

**Success Criteria:** Production-ready app with 82%+ accuracy, ready for defense

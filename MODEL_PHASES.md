# Model Training Phases Log

## Architecture
- **Base Model:** EfficientNet-B4 (ImageNet pretrained)
- **Input Resolution:** 380x380
- **Output Classes:** 24 (Phase 7)

## Phase Summary

| Phase | Strategy | Val Acc | Key Change |
|-------|----------|---------|------------|
| 1-2 | Frozen + unfreeze | ~55% | Baseline |
| 3 | Full unfreeze | 63.17% | LR scheduler |
| 4 | Rebalanced weights | 67.25% | Fixed bias |
| 5 | 380x380 resolution | 72% | Better texture |
| 6 | Extended training | 77% | Convergence |
| 7 | Clean 24 classes | TBD | Better data |

## Critical Bugs Fixed

### Bug 1: Double Rescaling (Caused 54.7% collapse)
- ImageDataGenerator(rescale=1./255) + EfficientNet's internal normalization
- **Fix:** Removed rescale, use raw [0,255] values
- **Impact:** Model recovered to functioning

### Bug 2: Aggressive Class Weights (Caused melanoma bias)
- Balanced weights with 20x ratio
- **Fix:** Sqrt-scaled weights, capped 1.8x
- **Impact:** Nevus→melanoma errors 152 → 72

### Bug 3: Kaggle Quick Save (Lost Phase 2)
- "Quick Save" doesn't persist working files
- **Fix:** Use "Commit" + immediate download
- **Impact:** All future checkpoints preserved

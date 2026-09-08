# Detailed Training Phases Breakdown

## Phase 1: Baseline (Head Training Only)

**Strategy:** Freeze entire EfficientNet-B4 base, train only dense head

```python
base_model.trainable = False
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
optimizer = Adam(learning_rate=0.001)
```

**Results:**
- Epochs: ~15
- Accuracy: ~50%
- Purpose: Let head learn initial feature discrimination

**Lessons:**
- Model learns very slowly (only dense layers training)
- But stable — no divergence

---

## Phase 2: Unfreeze Last 100 Layers

**Strategy:** Unfreeze last 100 layers of base_model, continue training

```python
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False
optimizer = Adam(learning_rate=0.00001)  # Much lower LR
```

**Results:**
- Epochs: ~60
- Accuracy: ~55%
- Improvement: +5% over Phase 1

**Lessons:**
- Unfreezing helps but slow progress
- Lower LR needed to not destroy pretrained weights

---

## Phase 3: Full Unfreeze + LR Scheduler

**Strategy:** Unfreeze entire model, use ReduceLROnPlateau

```python
base_model.trainable = True  # All layers trainable
optimizer = Adam(learning_rate=0.000005)
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7)
]
```

**Results:**
- Epochs: ~30
- Accuracy: **63.17%** ⬆️
- Improvement: +8% over Phase 2

**Lessons:**
- Full unfreeze + learning rate scheduler crucial
- Model finally breaking through 60% barrier
- LR reduction helps refine weights

---

## Phase 4: Rebalanced Class Weights (CRITICAL FIX)

**Problem Discovered:** Melanoma over-prediction bias
- Melanoma precision: 0.43 (many false positives)
- Nevus recall: 0.77 (missing true nevus)
- Nevus→melanoma confusion: **152 errors**

**Root Cause:** Aggressive balanced class weights (20x ratio)
- Nevus: 0.15 weight (common, penalized lightly)
- Melanoma: 3.0 weight (rare, penalized heavily)

**Fix Applied:**
```python
class_weights_arr = compute_class_weight('balanced', ...)
sqrt_weights = np.sqrt(class_weights_arr)  # Sqrt scaling
class_weight_dict = {k: min(v, 1.8) for k, v in dict(enumerate(sqrt_weights)).items()}
# Result: ~4-5x ratio instead of ~20x
```

**Results:**
- Epochs: ~30
- Accuracy: **67.25%** ⬆️
- Melanoma precision: 0.43 → 0.53 ✅
- Nevus recall: 0.77 → 0.88 ✅
- Nevus→melanoma errors: 152 → 72 ✅

**Lessons:**
- Class weights matter MORE than other hyperparameters
- Sqrt-scaling is gentler than full balanced weights
- Cap at 1.8x prevents extreme bias

---

## Phase 5: Higher Resolution (380x380)

**Motivation:** EfficientNet-B4 native resolution is 380x380, not 224x224

**Strategy:**
```python
IMG_SIZE = 380  # was 224
# Rebuild architecture at new input_shape
base_model = EfficientNetB4(weights=None, input_shape=(380, 380, 3))
model.load_weights(phase4_checkpoint)  # WEIGHTS ONLY, not full model
```

**Why This Works:**
- Conv kernels are resolution-independent
- Only base_model's input_shape metadata changes
- Dense layers unchanged
- Can load Phase 4 weights directly into new architecture

**Results:**
- Epochs: ~30
- Accuracy: **72%** ⬆️
- Improvement: +5% from Phase 4

**Lessons:**
- 380x380 = 2.9x more pixels than 224x224
- Batch size reduced 32→16 (memory constraint)
- Better texture detail helps distinguish similar conditions

---

## Phase 6: Class Merge (29 classes) + Extended Training

**Motivation:** DermNet "Melanoma Skin Cancer Nevi and Moles" folder contains clean split of melanoma vs nevus — merge into HAM10000 classes

**Changes:**
```python
# Remove DermNet melanoma/nevus folder (now merged)
# Reduce classes: 30 → 29
# Re-initialize final Dense(29) layer
# Load Phase 5 backbone + dense head (skip mismatched final layer)
```

**Results:**
- Initial: 72.12% (Phase 5 checkpoint, 30 classes)
- Final at epoch 25: **74.88%** ⬆️
- Merge reduced false positives (cleaner training)
- Early stopping triggered: session timeout before convergence

**Lessons:**
- Class merging can improve accuracy when justified
- Sklearn's `by_name=True, skip_mismatch=True` useful for architecture changes
- Document all class decisions (why merge this one, not others)

---

## Phase 7: Clean 24-Class Dataset (CURRENT)

**Motivation:** Confusion matrix showed 12 classes with F1 < 0.65 (weak performers)

**Analysis (29 classes):**

Top 5 worst:

Cellulitis/Impetigo (F1: 0.407, 52 samples)
Herpes/HPV (F1: 0.550, 80 samples)
Poison Ivy (F1: 0.554, 46 samples)
Lupus (F1: 0.556, 88 samples)
actinic_keratosis (F1: 0.571, 47 samples)

**Decision:** Remove 5 worst, keep 24 best classes

**Strategy:**
```python
# Fresh training from ImageNet weights (not Phase 6 checkpoint)
base_model = EfficientNetB4(weights='imagenet', input_shape=(380, 380, 3))
# Train on clean 24-class dataset
# Expected: cleaner learning without weak classes dragging down
```

**Expected Results:**
- Accuracy: **82-85%** ⬆️⬆️
- Reason: Fewer confusing classes = better model focus
- Training time: 8-10 hours on Kaggle GPU

**Lessons:**
- Data quality > model complexity
- Removing weak classes can improve overall accuracy
- Fresh start with clean data often better than fine-tuning dirty data

---

## Summary Table

| Phase | Res | Classes | LR | Key Change | Accuracy | Improvement |
|-------|-----|---------|-----|------------|----------|------------|
| 1 | 224 | 7 | 1e-3 | Frozen base | ~50% | — |
| 2 | 224 | 7 | 1e-5 | Unfreeze 100 | ~55% | +5% |
| 3 | 224 | 7 | 5e-6 | Full unfreeze | 63.17% | +8% |
| 4 | 224 | 7 | 1e-5 | Fixed weights | 67.25% | +4% |
| 5 | 380 | 7 | 8e-6 | Higher res | 72% | +5% |
| 6 | 380 | 29 | 3e-5 | Class merge | 74.88% | +2.88% |
| 7 | 380 | 24 | 1e-5 | Clean data | TBD | +5-8%? |

---

## Key Takeaways

1. **Class weights are critical** — Phase 4 fix was game-changer
2. **Resolution matters** — 380x380 better than 224x224 for texture
3. **Extended training helps** — Phase 6 showed steady improvement
4. **Data quality > quantity** — Phase 7 removes weak classes for focus
5. **Learning rate crucial** — Must lower when unfreezing pretrained layers

---

## Supporting Scripts

The following analysis and utility scripts document the training pipeline:

### `04_inspect_overlaps.py`
Analyzes DermNet folders that appear to overlap with HAM10000 classes.
- Counts images in potential overlap folders
- Extracts and counts filename prefixes
- Identifies which overlaps are "clean splits" vs "grab-bags"
- **Decision:** Only merge "Melanoma Skin Cancer Nevi and Moles" (clean split)

### `05_merge_dermnet_melanoma_nevus.py`
Implements the merge of DermNet melanoma/nevus into HAM10000 classes.
- Splits by filename prefix (e.g., "malignant-melanoma" → melanoma class)
- Excludes ambiguous cases (melanotic-macule, mongolian-spot)
- Reduces class count from 30 → 29

**Result:**

train: +111 melanoma, +282 nevus
val: +27 melanoma, +56 nevus
test: +21 melanoma, +68 nevus


### `phase5_analyze.py`
Generates confusion matrix and texture-cluster analysis post-training.
- Full classification report (precision, recall, F1 per class)
- Full confusion matrix heatmap (30x30)
- Zoomed texture-cluster heatmap (5 classes: Psoriasis, Eczema, Tinea, Nail Fungus, Scabies)

### `verify_checkpoint.py`
Validation-only script to verify a checkpoint's accuracy without training.
- Rebuilds architecture from scratch
- Loads weights only
- Runs evaluation (no model.fit)
- **Use case:** Confirm downloaded .h5 file integrity

**Example output (Phase 5.2):**

✅ Verified val_accuracy: 0.7212 (72.12%)
Verified val_loss: 1.2146

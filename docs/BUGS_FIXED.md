# Critical Bugs Found & Fixed

## Bug 1: Double Rescaling (SEVERITY: CRITICAL) 🔴

### The Problem
Model collapsed to predicting single class with 54.7% accuracy (guessing majority class)

### Root Cause
```python
# WRONG: Both rescaling + internal normalization
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescales to [0, 1]
# EfficientNet-B4 expects [0, 255] and rescales internally
# Result: Double rescaling → values near zero → no signal
```

### Impact
- Accuracy dropped from normal range to single-class prediction
- Model completely non-functional regardless of hyperparameter tuning
- Wasted ~10 hours of GPU time before detection

### The Fix
```python
# CORRECT: No rescale, let EfficientNet handle normalization
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    # NO rescale parameter!
    horizontal_flip=True,
    fill_mode='nearest'
)
# EfficientNet-B4 will rescale internally
```

### How It Was Discovered
- Phase 2 accuracy stuck at 54.7% across all epochs
- Checked data: images were near-black (values ~0-1 instead of 0-255)
- Realized rescaling was happening twice
- Removed rescale parameter → accuracy recovered to 55%+

### Lesson Learned
**Always check what preprocessing the base model expects!**
- EfficientNet: Expects [0, 255], rescales internally
- MobileNet: Expects [0, 255], rescales internally
- ResNet: Expects [-1, 1] or [0, 1] depending on source

---

## Bug 2: Aggressive Class Weights (SEVERITY: HIGH) 🟠

### The Problem
Model had strong melanoma over-prediction bias:
- Melanoma precision: 0.43 (56% false positive rate)
- Nevus recall: 0.77 (23% of true nevus missed)
- Nevus→melanoma misclassification: **152 errors** out of 1,500 test images

### Root Cause
```python
# WRONG: Using full balanced weights with high cap
class_weights_arr = compute_class_weight('balanced', ...)
class_weight_dict = {k: min(v, 2.0) for k, v in dict(enumerate(class_weights_arr)).items()}

# Result: nevus weight ≈ 0.15, melanoma weight ≈ 3.0
# Ratio: 3.0 / 0.15 = 20x difference
# Model learned: "Predict melanoma when unsure — penalty is 20x higher"
```

### Impact
- Clinically dangerous: missing melanoma is serious, but over-predicting is also bad
- 152 false melanoma predictions per 1,500 images
- Decision boundary heavily skewed toward melanoma

### The Fix
```python
# CORRECT: Sqrt-scaled weights, capped at 1.8x ratio
class_weights_arr = compute_class_weight('balanced', classes=..., y=...)
sqrt_weights = np.sqrt(class_weights_arr)  # Gentler scaling
class_weight_dict = {k: min(v, 1.8) for k, v in dict(enumerate(sqrt_weights)).items()}

# Result: nevus weight ≈ 0.4, melanoma weight ≈ 1.8
# Ratio: 1.8 / 0.4 = 4.5x difference (much more balanced)
```

### Results After Fix
- Melanoma precision: 0.43 → 0.53 (+23% improvement) ✅
- Nevus recall: 0.77 → 0.88 (+14% improvement) ✅
- Nevus→melanoma errors: 152 → 72 (-53% reduction) ✅
- Overall accuracy: 67.25% (+4% from Phase 4)

### Confusion Matrix Change

BEFORE (aggressive weights):
Predicted
True melanoma nevus other
mel 74 42 15 (melanoma recall: 62%)
nev 152 1025 178 (nevus →mel: 152 errors!)
oth ...

AFTER (sqrt-scaled weights):
Predicted
True melanoma nevus other
mel 89 30 12 (melanoma recall: 74%)
nev 72 1211 72 (nevus →mel: 72 errors)
oth ...


### Lesson Learned
**Class weights amplify bias — use sqrt scaling instead of full balanced**
- Full balanced: weight_rare / weight_common = high ratio
- Sqrt balanced: sqrt(weight_rare / weight_common) = gentler
- Always cap at reasonable ratio (1.5-2.0x) to prevent extreme bias

---

## Bug 3: Kaggle "Quick Save" Unreliable (SEVERITY: HIGH) 🟠

### The Problem
Lost entire Phase 2 checkpoint (60 epochs of training, ~6 hours on GPU)

### What Happened
1. Trained Phase 2 for 60 epochs
2. Clicked "Save" in Kaggle notebook
3. Checked later: `/kaggle/working/` files missing
4. Weeks of work lost because checkpoint wasn't recovered

### Root Cause
- Kaggle "Quick Save" is temporary session-only storage
- `/kaggle/working/` persists only during active session
- After session ends or notebook restarts: files gone unless committed
- Must use "Save & Run All (Commit)" to persist files

### The Fix
```bash
# WRONG: Relying on Quick Save
# Model saved in /kaggle/working/
# Session ends → files deleted

# CORRECT: Save → Commit → Immediately Download
1. Model saves to /kaggle/working/model.h5
2. Click "Save & Run All (Commit)" button
3. Files are committed to Kaggle dataset
4. Immediately download from Output tab
5. Save locally or to Google Drive
```

### How to Prevent
- Never rely on Kaggle "Quick Save" alone
- After important training run:
  1. Click **"Save & Run All (Commit)"**
  2. Wait for commit confirmation
  3. Go to **"Output"** tab
  4. **Download** the .h5 file immediately
  5. Upload to Google Drive as backup
- Use `TimeLimitCallback` to stop before session timeout

### Lesson Learned
**Always back up critical files locally or to cloud**
- Kaggle: use Commit + immediate download
- Colab: save to Google Drive (`/content/drive/...`)
- GitHub: commit code scripts (but not .h5 files)

---

## Bug 4: HAM10000 Images in Nested Folders (SEVERITY: MEDIUM) 🟡

### The Problem
`prepare_data.py` couldn't find HAM10000 images

### What Happened
1. Downloaded HAM10000 from Kaggle
2. Extracted to `D:\FYP\ml\data\ham10000\`
3. But images were in subfolders:

ham10000/
├── HAM10000_images_part_1/
│ ├── ISIC_0000000.jpg
│ └── ...
├── HAM10000_images_part_2/
│ ├── ISIC_0005000.jpg
│ └── ...
└── HAM10000_metadata.csv

4. Script looked for `ham10000/*.jpg` → found 0 images

### The Fix
```python
# WRONG: Direct glob
for image_id in ham_metadata.keys():
    src = f"{HAM_PATH}/{image_id}.jpg"  # Fails if nested

# CORRECT: Recursive search
import os
for root, dirs, files in os.walk(HAM_PATH):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            # Process image
```

### Lesson Learned
**Always use recursive search for dataset files, they're often nested**

---

## Summary Table

| Bug | Severity | Cause | Fix | Impact |
|-----|----------|-------|-----|--------|
| Double Rescaling | Critical | EfficientNet internal rescale | Removed rescale param | Accuracy recovered |
| Class Weights | High | 20x weight ratio | Sqrt-scale, cap 1.8x | +4% accuracy |
| Kaggle Save | High | Quick Save not persistent | Use Commit + download | Checkpoints preserved |
| Nested Folders | Medium | Subfolder structure | Recursive os.walk() | Found all images |

---

## Prevention Checklist for Future

- [ ] Verify data preprocessing matches base model expectations
- [ ] Check class weight ratios (should be 1.5-2.0x max)
- [ ] Always commit + download from Kaggle (never rely on Quick Save)
- [ ] Use recursive search for dataset files
- [ ] Log confusion matrix per phase to catch bias early
- [ ] Save checkpoints to both local and cloud storage

# ============================================================
# EpiTrack: Inspect overlapping DermNet/HAM10000 classes
# Run this BEFORE the merge script (05) to understand what's actually
# inside the DermNet catch-all folders that appear to duplicate HAM classes.
# ============================================================
import os
from collections import Counter

ORGANIZED_DATA = "/kaggle/input/datasets/azemmeer/epitrack/organized"
TRAIN_PATH = f"{ORGANIZED_DATA}/train"

# ---- Step 1: basic overlap counts ----
overlap_pairs = {
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": ["actinic_keratosis", "basal_cell_carcinoma"],
    "Melanoma Skin Cancer Nevi and Moles": ["melanoma", "nevus"],
    "Seborrheic Keratoses and other Benign Tumors": ["benign_keratosis", "dermatofibroma"],
}

for dermnet_class, ham_classes in overlap_pairs.items():
    path = os.path.join(TRAIN_PATH, dermnet_class)
    n_images = len(os.listdir(path)) if os.path.exists(path) else "MISSING"
    print(f"\n{dermnet_class}: {n_images} images")
    for f in os.listdir(path)[:5]:
        print(f"  sample: {f}")
    for ham_class in ham_classes:
        hp = os.path.join(TRAIN_PATH, ham_class)
        n = len(os.listdir(hp)) if os.path.exists(hp) else "MISSING"
        print(f"  vs {ham_class}: {n} images")

# ---- Step 2: filename-prefix breakdown per folder ----
folders_to_inspect = [
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Melanoma Skin Cancer Nevi and Moles",
    "Seborrheic Keratoses and other Benign Tumors",
]

for folder in folders_to_inspect:
    path = os.path.join(TRAIN_PATH, folder)
    files = os.listdir(path)
    prefixes = Counter()
    for f in files:
        name = f.rsplit('.', 1)[0]
        parts = name.rsplit('-', 1)
        prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
        prefixes[prefix] += 1

    print(f"\n{'='*60}")
    print(f"{folder} — {len(files)} images, {len(prefixes)} distinct prefixes")
    print('='*60)
    for prefix, count in prefixes.most_common(30):
        print(f"  {count:>4}  {prefix}")

# ============================================================
# FINDINGS:
#
# - "Actinic Keratosis Basal Cell Carcinoma..." (1016 images, 59 prefixes):
#   genuine grab-bag — NOT a clean 2-way split. Left as its own class.
#
# - "Melanoma Skin Cancer Nevi and Moles" (404 images, 16 prefixes):
#   CLEAN split — nearly every prefix maps unambiguously to either
#   melanoma or nevus. This is the one merged (see script 05).
#
# - "Seborrheic Keratoses and other Benign Tumors" (1206 images, 63 prefixes):
#   genuine grab-bag. NOT a clean 2-way split. Left as its own class.
# ============================================================

# ============================================================
# EpiTrack: Merge DermNet "Melanoma Skin Cancer Nevi and Moles"
# into HAM10000's melanoma / nevus classes
#
# Result (actual run against this dataset):
#   train: +111 melanoma, +282 nevus, 11 excluded, 0 unmatched
#   val:   +27 melanoma,  +56 nevus,   0 excluded, 0 unmatched
#   test:  +21 melanoma,  +68 nevus,   3 excluded, 0 unmatched
#   -> 30 classes reduced to 29 (DermNet melanoma/nevus folder emptied out)
# ============================================================

import os
import shutil

BASE = "/kaggle/working/organized_merged"  # write to new location
ORIGINAL = "/kaggle/input/datasets/azemmeer/epitrack/organized"

SOURCE_FOLDER = "Melanoma Skin Cancer Nevi and Moles"

# prefix -> target class (derived from script 04's prefix breakdown)
MELANOMA_PREFIXES = {"malignant-melanoma", "lentigo-maligna"}
NEVUS_PREFIXES = {
    "melanocytic-nevi", "congenital-nevus", "atypical-nevi-dermoscopy",
    "atypical-nevi", "atypical-nevi-trunk", "halo-nevus", "becker-nevus",
    "nevus-spilus", "blue-nevus", "blue-nevus-ota", "spitz-nevus",
    "nevus-repigmented"
}
# Neither melanoma nor a true nevus — excluded
EXCLUDE_PREFIXES = {"melanotic-macule", "mongolian-spot"}

def get_prefix(filename):
    name = filename.rsplit('.', 1)[0]
    parts = name.rsplit('-', 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else name

for split in ["train", "val", "test"]:
    print(f"\n{'='*60}\nProcessing split: {split}\n{'='*60}")
    split_src = os.path.join(ORIGINAL, split)
    split_dst = os.path.join(BASE, split)

    # First, copy every OTHER class folder unchanged
    for cls in os.listdir(split_src):
        if cls == SOURCE_FOLDER:
            continue
        src_cls_path = os.path.join(split_src, cls)
        dst_cls_path = os.path.join(split_dst, cls)
        if not os.path.exists(dst_cls_path):
            shutil.copytree(src_cls_path, dst_cls_path)

    # Now split the DermNet melanoma/nevus folder
    source_path = os.path.join(split_src, SOURCE_FOLDER)
    if not os.path.exists(source_path):
        print(f"  ⚠️ {SOURCE_FOLDER} not found in {split}, skipping")
        continue

    dst_melanoma = os.path.join(split_dst, "melanoma")
    dst_nevus = os.path.join(split_dst, "nevus")
    os.makedirs(dst_melanoma, exist_ok=True)
    os.makedirs(dst_nevus, exist_ok=True)

    counts = {"melanoma": 0, "nevus": 0, "excluded": 0, "unmatched": 0}

    for fname in os.listdir(source_path):
        prefix = get_prefix(fname)
        src_file = os.path.join(source_path, fname)

        if prefix in MELANOMA_PREFIXES:
            shutil.copy2(src_file, os.path.join(dst_melanoma, f"dermnet_{fname}"))
            counts["melanoma"] += 1
        elif prefix in NEVUS_PREFIXES:
            shutil.copy2(src_file, os.path.join(dst_nevus, f"dermnet_{fname}"))
            counts["nevus"] += 1
        elif prefix in EXCLUDE_PREFIXES:
            counts["excluded"] += 1
        else:
            print(f"  ⚠️ Unmatched prefix '{prefix}' for file: {fname}")
            counts["unmatched"] += 1

    print(f"  → melanoma: +{counts['melanoma']}")
    print(f"  → nevus: +{counts['nevus']}")
    print(f"  → excluded: {counts['excluded']}")
    print(f"  → unmatched (needs review): {counts['unmatched']}")

print("\n" + "="*60)
print("DONE. Final class counts:")
print("="*60)
for split in ["train", "val", "test"]:
    split_dst = os.path.join(BASE, split)
    n_classes = len(os.listdir(split_dst))
    print(f"\n{split}: {n_classes} classes")
    for cls in sorted(os.listdir(split_dst)):
        n = len(os.listdir(os.path.join(split_dst, cls)))
        print(f"  {cls}: {n}")

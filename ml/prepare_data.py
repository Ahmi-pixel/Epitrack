"""
EpiTrack: Data Preparation (HAM10000 + DermNet)
================================================
Combines HAM10000 (7 skin cancer classes) and DermNet (23 general
dermatology classes) into a single organized train/val/test split
covering 30 disease classes total.

Split ratio: 70% train / 15% val / 15% test, applied per-class so
each class is represented proportionally in every split.

Usage:
    python prepare_data.py

Expects:
    <HAM10000_PATH>/HAM10000_metadata.csv
    <HAM10000_PATH>/*.jpg (images, possibly nested in
        HAM10000_images_part_1/ and HAM10000_images_part_2/ subfolders)
    <DERMNET_PATH>/train/<class_name>/*.jpg
    <DERMNET_PATH>/test/<class_name>/*.jpg

Outputs:
    <OUTPUT_PATH>/train/<class_name>/*.jpg
    <OUTPUT_PATH>/val/<class_name>/*.jpg
    <OUTPUT_PATH>/test/<class_name>/*.jpg
"""

import os
import shutil
import random
import pandas as pd

# ============================================================
# Paths — adjust these for your environment
# ============================================================
HAM10000_PATH = r"D:\FYP\ml\data\ham10000"
DERMNET_TRAIN_PATH = r"D:\FYP\ml\data\Dermnet\train"
DERMNET_TEST_PATH = r"D:\FYP\ml\data\Dermnet\test"
OUTPUT_PATH = r"D:\FYP\ml\data\organized"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # reproducible splits

# HAM10000 diagnosis code -> readable class name
HAM10000_CLASS_MAP = {
    "akiec": "actinic_keratosis",
    "bcc": "basal_cell_carcinoma",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "nevus",
    "vasc": "vascular_lesion",
}


def find_ham10000_images(ham10000_path):
    """
    Build a filename -> full path index by walking the HAM10000 directory
    recursively. Images are sometimes nested inside
    HAM10000_images_part_1/ and HAM10000_images_part_2/ subfolders
    depending on how the Kaggle zip was extracted, so we search
    recursively rather than assuming a flat structure.
    """
    index = {}
    for root, _dirs, files in os.walk(ham10000_path):
        for f in files:
            if f.lower().endswith(".jpg"):
                index[f] = os.path.join(root, f)
    return index


def read_ham10000_metadata(ham10000_path):
    print("\n📖 Reading HAM10000 metadata...")
    metadata_csv = os.path.join(ham10000_path, "HAM10000_metadata.csv")
    df = pd.read_csv(metadata_csv)
    print(f"✅ Found {len(df)} HAM10000 images")

    print("\nHAM10000 Disease Distribution:")
    for code in sorted(df["dx"].unique()):
        count = (df["dx"] == code).sum()
        print(f"  {code}: {count}")

    return df


def scan_dermnet_classes(dermnet_train_path, dermnet_test_path):
    print("\n📂 Scanning DermNet disease folders...")
    class_names = sorted(os.listdir(dermnet_train_path))
    class_names = [c for c in class_names
                   if os.path.isdir(os.path.join(dermnet_train_path, c))]

    print(f"✅ Found {len(class_names)} DermNet disease categories:")
    class_files = {}
    for class_name in class_names:
        train_dir = os.path.join(dermnet_train_path, class_name)
        test_dir = os.path.join(dermnet_test_path, class_name)

        files = []
        for d in (train_dir, test_dir):
            if os.path.isdir(d):
                files.extend(
                    os.path.join(d, f) for f in os.listdir(d)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
        class_files[class_name] = files
        print(f"  {class_name}: {len(files)} images")

    return class_files


def split_files(files, train_ratio, val_ratio):
    """Shuffle and split a list of file paths into train/val/test."""
    files = list(files)
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def copy_split(split_files_dict, class_name, output_path):
    counts = {}
    for split_name, files in split_files_dict.items():
        dest_dir = os.path.join(output_path, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for src_path in files:
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
        counts[split_name] = len(files)
    return counts


def main():
    print("=" * 60)
    print("EpiTrack: Data Preparation (HAM10000 + DermNet)")
    print("=" * 60)
    print(f"\nHAM10000 path: {HAM10000_PATH}")
    print(f"DermNet train path: {DERMNET_TRAIN_PATH}")
    print(f"DermNet test path: {DERMNET_TEST_PATH}")
    print(f"Output path: {OUTPUT_PATH}")

    # --- HAM10000 ---
    ham_df = read_ham10000_metadata(HAM10000_PATH)
    ham_image_index = find_ham10000_images(HAM10000_PATH)

    # --- DermNet ---
    dermnet_class_files = scan_dermnet_classes(DERMNET_TRAIN_PATH, DERMNET_TEST_PATH)

    # --- Create folder structure ---
    print("\n📁 Creating folder structure...")
    all_classes = list(HAM10000_CLASS_MAP.values()) + list(dermnet_class_files.keys())
    for split in ("train", "val", "test"):
        for class_name in all_classes:
            os.makedirs(os.path.join(OUTPUT_PATH, split, class_name), exist_ok=True)
    print(f"✅ Total disease classes: {len(all_classes)}")
    print("✅ Folder structure created!")

    # --- Copy HAM10000 images ---
    print("\n📸 Copying HAM10000 images...")
    missing = 0
    ham_by_class = {code: [] for code in HAM10000_CLASS_MAP}
    for _, row in ham_df.iterrows():
        filename = row["image_id"] + ".jpg"
        dx = row["dx"]
        if filename in ham_image_index:
            ham_by_class[dx].append(ham_image_index[filename])
        else:
            missing += 1

    print(f"Found {sum(len(v) for v in ham_by_class.values())} HAM10000 images")
    if missing:
        print(f"⚠️  {missing} images listed in metadata but not found on disk")

    ham_totals = {"train": 0, "val": 0, "test": 0}
    for dx, class_name in HAM10000_CLASS_MAP.items():
        splits = split_files(ham_by_class[dx], TRAIN_RATIO, VAL_RATIO)
        counts = copy_split(splits, class_name, OUTPUT_PATH)
        for split_name, count in counts.items():
            ham_totals[split_name] += count

    print(f"Copying to train ({ham_totals['train']} images)...")
    print(f"Copying to val ({ham_totals['val']} images)...")
    print(f"Copying to test ({ham_totals['test']} images)...")
    print("✅ HAM10000 images copied!")

    # --- Copy DermNet images ---
    print("\n📸 Copying DermNet images...")
    dermnet_total = sum(len(files) for files in dermnet_class_files.values())
    print(f"Found {dermnet_total} DermNet images")

    dermnet_totals = {"train": 0, "val": 0, "test": 0}
    for class_name, files in dermnet_class_files.items():
        splits = split_files(files, TRAIN_RATIO, VAL_RATIO)
        counts = copy_split(splits, class_name, OUTPUT_PATH)
        for split_name, count in counts.items():
            dermnet_totals[split_name] += count

    print(f"Copying to train ({dermnet_totals['train']} images)...")
    print(f"Copying to val ({dermnet_totals['val']} images)...")
    print(f"Copying to test ({dermnet_totals['test']} images)...")
    print("✅ DermNet images copied!")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 60)

    print("\n📊 Final Image Count:")
    for split in ("train", "val", "test"):
        split_dir = os.path.join(OUTPUT_PATH, split)
        total = 0
        counts = {}
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            n = len(os.listdir(class_dir))
            counts[class_name] = n
            total += n
        print(f"\n{split.upper()} ({total} total):")
        for class_name, n in counts.items():
            print(f"  {class_name}: {n}")

    print("\n✅ All data organized and ready for training!")


if __name__ == "__main__":
    main()

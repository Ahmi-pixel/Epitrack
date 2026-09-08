# ============================================================
# EpiTrack Phase 1: Baseline Training (Head Only)
# Strategy: Freeze entire EfficientNet-B4 base, train only dense head
# Expected: ~50% accuracy on 7 HAM10000 classes
# ============================================================

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = r"D:\FYP\ml\data\organized\train"
VAL_PATH = r"D:\FYP\ml\data\organized\val"
MODEL_SAVE_PATH = r"D:\FYP\ml\models\efficientnet_b4_phase1.h5"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

print("=" * 60)
print("EpiTrack: Phase 1 — Baseline (Head Training Only)")
print("=" * 60)

# ============================================================
# DATA GENERATORS
# ============================================================

print("\nPreparing data...")

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

NUM_CLASSES = len(train_data.class_indices)
print(f"✅ {NUM_CLASSES} classes")
print(f"✅ {train_data.samples} training images")
print(f"✅ {val_data.samples} validation images")

# ============================================================
# BUILD MODEL (FROZEN BASE)
# ============================================================

print("\nBuilding model...")

base_model = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# FREEZE entire base
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled (base frozen, head trainable only)")

# ============================================================
# TRAIN
# ============================================================

print(f"\n🚀 Training Phase 1 ({EPOCHS} epochs, frozen base)...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    verbose=1
)

# ============================================================
# SAVE & EVALUATE
# ============================================================

print(f"\nSaving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n" + "=" * 60)
print("PHASE 1 RESULTS")
print("=" * 60)
print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Model saved: {MODEL_SAVE_PATH}")
print("=" * 60)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Phase 1: Accuracy (Frozen Base)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Phase 1: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"D:\FYP\ml\models\phase1_history.png", dpi=100)
print("\n✅ Training plot saved")

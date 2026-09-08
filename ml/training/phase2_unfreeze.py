# ============================================================
# EpiTrack Phase 2: Unfreeze Last 100 Layers
# Strategy: Unfreeze last 100 layers of EfficientNet-B4, fine-tune
# Expected: ~55% accuracy
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
PHASE1_CHECKPOINT = r"D:\FYP\ml\models\efficientnet_b4_phase1.h5"
MODEL_SAVE_PATH = r"D:\FYP\ml\models\efficientnet_b4_phase2.h5"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60

print("=" * 60)
print("EpiTrack: Phase 2 — Unfreeze Last 100 Layers")
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

# ============================================================
# LOAD PHASE 1 & UNFREEZE
# ============================================================

print("\nLoading Phase 1 checkpoint...")
model = tf.keras.models.load_model(PHASE1_CHECKPOINT)

base_model = model.layers[0]
print(f"Total base layers: {len(base_model.layers)}")

# Unfreeze last 100 layers
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

print(f"Unfrozen layers: {len(base_model.layers[-100:])}")

# ============================================================
# RECOMPILE (MUCH LOWER LR)
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Much lower!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model recompiled (last 100 layers unfrozen, LR=1e-5)")

# ============================================================
# TRAIN
# ============================================================

print(f"\n🚀 Training Phase 2 ({EPOCHS} epochs, last 100 unfrozen)...\n")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# RESULTS
# ============================================================

best_val_acc = max(history.history['val_accuracy'])

print("\n" + "=" * 60)
print("PHASE 2 RESULTS")
print("=" * 60)
print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Improvement from Phase 1: +{(best_val_acc - 0.50)*100:.2f}%")
print(f"Model saved: {MODEL_SAVE_PATH}")
print("=" * 60)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.axhline(y=0.50, color='r', linestyle='--', label='Phase 1 (50%)')
plt.title('Phase 2: Accuracy (Unfreeze Last 100)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Phase 2: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"D:\FYP\ml\models\phase2_history.png", dpi=100)
print("\n✅ Training plot saved")

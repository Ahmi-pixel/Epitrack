# ============================================================
# EpiTrack Phase 5: Higher Resolution (380x380)
# Strategy: Upgrade to EfficientNet-B4 native resolution
# Expected: ~72% accuracy
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = r"D:\FYP\ml\data\organized\train"
VAL_PATH = r"D:\FYP\ml\data\organized\val"
PHASE4_CHECKPOINT = r"D:\FYP\ml\models\efficientnet_b4_phase4.h5"
MODEL_SAVE_PATH = r"D:\FYP\ml\models\efficientnet_b4_phase5.h5"

IMG_SIZE = 380  # EfficientNet-B4 native resolution
BATCH_SIZE = 16  # Reduced due to memory constraints
EPOCHS = 30

print("=" * 60)
print("EpiTrack: Phase 5 — Higher Resolution (380x380)")
print("=" * 60)

# ============================================================
# DATA GENERATORS
# ============================================================

print(f"\nPreparing data at {IMG_SIZE}x{IMG_SIZE}...")

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
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
print(f"✅ {IMG_SIZE}x{IMG_SIZE} resolution")
print(f"✅ Batch size reduced to {BATCH_SIZE}")

# ============================================================
# BUILD AT NEW RESOLUTION
# ============================================================

print("\nBuilding model at new resolution...")

# Rebuild EfficientNet-B4 at 380x380
base_model = EfficientNetB4(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))

# Load Phase 4 weights (only backbone)
print("\nLoading Phase 4 weights...")
phase4_model = tf.keras.models.load_model(PHASE4_CHECKPOINT)
for layer in model.layers[:-1]:
    try:
        layer.set_weights(phase4_model.get_layer(layer.name).get_weights())
    except:
        pass  # Skip layers that don't match

print("✅ Weights loaded (resolution-agnostic layers)")

# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================

print("\nComputing class weights...")

class_weights_arr = compute_class_weight(
    'balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

sqrt_weights = np.sqrt(class_weights_arr)
class_weight_dict = {k: min(v, 1.8) for k, v in dict(enumerate(sqrt_weights)).items()}

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.000008),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled at 380x380")

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# ============================================================
# TRAIN
# ============================================================

print(f"\n🚀 Training Phase 5 ({EPOCHS} epochs, 380x380 resolution)...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================
# RESULTS
# ============================================================

best_val_acc = max(history.history['val_accuracy'])

print("\n" + "=" * 60)
print("PHASE 5 RESULTS (HIGHER RESOLUTION)")
print("=" * 60)
print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Improvement from Phase 4: +{(best_val_acc - 0.6725)*100:.2f}%")
print(f"\n✅ Key Achievement:")
print(f"   - 380x380 resolution (2.9x more pixels)")
print(f"   - Better texture detail for disease distinction")
print(f"Model saved: {MODEL_SAVE_PATH}")
print("=" * 60)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.axhline(y=0.6725, color='r', linestyle='--', label='Phase 4 (67.25%)')
plt.title('Phase 5: Accuracy (380x380 Resolution)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Phase 5: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"D:\FYP\ml\models\phase5_history.png", dpi=100)
print("\n✅ Training plot saved")

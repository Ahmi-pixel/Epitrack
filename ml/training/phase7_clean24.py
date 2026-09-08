# ============================================================
# EpiTrack Phase 7: Clean 24-Class Dataset Retraining
# Strategy: Remove 5 weakest classes, retrain on 24 best classes
# Expected: ~82-85% accuracy
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight
import time
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# For Kaggle/Colab — update paths accordingly
TRAIN_PATH = "/kaggle/input/datasets/azemmeer/epitrack-clean-24-classes/organized_clean_24/train"
VAL_PATH = "/kaggle/input/datasets/azemmeer/epitrack-clean-24-classes/organized_clean_24/val"
MODEL_SAVE_PATH = "/kaggle/working/efficientnet_b4_phase7_best.h5"
FINAL_MODEL_PATH = "/kaggle/working/efficientnet_b4_phase7_final.h5"

IMG_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 60

print("=" * 60)
print("EpiTrack: Phase 7 — Clean 24-Class Dataset Retraining")
print("=" * 60)

# ============================================================
# DATA GENERATORS
# ============================================================

print(f"\nPreparing clean 24-class dataset...")

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
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
print(f"✅ {NUM_CLASSES} disease classes (removed 5 weakest)")
print(f"✅ {train_data.samples} training images")
print(f"✅ {val_data.samples} validation images")

# ============================================================
# BUILD MODEL (FRESH FROM IMAGENET)
# ============================================================

print("\nBuilding model from ImageNet weights...")

base_model = EfficientNetB4(
    weights='imagenet',
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

print("✅ Class weights computed (sqrt-scaled)")

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled")

# ============================================================
# TIME LIMIT CALLBACK
# ============================================================

class TimeLimitCallback(Callback):
    def __init__(self, max_hours=7.5):
        super().__init__()
        self.max_seconds = max_hours * 3600
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"\n⏰ Time limit reached — stopping safely")
            self.model.stop_training = True

time_limit = TimeLimitCallback(max_hours=7.5)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    time_limit
]

# ============================================================
# TRAIN
# ============================================================

print(f"\n🚀 Training Phase 7 ({EPOCHS} epochs, clean 24-class dataset)...\n")
print("Expected: 82-85% accuracy on cleaner data\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================
# SAVE & EVALUATE
# ============================================================

print(f"\nSaving final model...")
model.save(FINAL_MODEL_PATH)

best_val_acc = max(history.history['val_accuracy'])
total_epochs = len(history.history['val_accuracy'])

print("\n" + "=" * 60)
print("PHASE 7 RESULTS (CLEAN 24-CLASS DATASET)")
print("=" * 60)
print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Improvement from Phase 6: +{(best_val_acc - 0.77)*100:.2f}%")
print(f"Total epochs: {total_epochs}")
print(f"\n✅ Key Achievements:")
print(f"   - Removed 5 weakest classes (F1 < 0.57)")
print(f"   - Fresh training from ImageNet weights")
print(f"   - Cleaner data = better convergence")
print(f"   - Ready for TFLite conversion!")
print(f"\nModels saved:")
print(f"  Best: {MODEL_SAVE_PATH}")
print(f"  Final: {FINAL_MODEL_PATH}")
print("=" * 60)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.axhline(y=0.77, color='r', linestyle='--', label='Phase 6 (77%)')
plt.axhline(y=0.80, color='g', linestyle='--', alpha=0.5, label='Target (80%)')
plt.title('Phase 7: Accuracy (Clean 24-Class Dataset)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Phase 7: Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/phase7_history.png', dpi=150)
print("\n✅ Training plot saved")

plt.show()

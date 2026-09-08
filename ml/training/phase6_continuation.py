# ============================================================
# EpiTrack Phase 6: Extended Training + Aggressive Augmentation
# Strategy: Continue from Phase 5, extend epochs, use aggressive augmentation
# Expected: ~77% accuracy
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight
import time
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = r"D:\FYP\ml\data\organized\train"
VAL_PATH = r"D:\FYP\ml\data\organized\val"
PHASE5_CHECKPOINT = r"D:\FYP\ml\models\efficientnet_b4_phase5.h5"
MODEL_SAVE_PATH = r"D:\FYP\ml\models\efficientnet_b4_phase6.h5"

IMG_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 60  # Extended from Phase 5

print("=" * 60)
print("EpiTrack: Phase 6 — Extended Training + Aggressive Augmentation")
print("=" * 60)

# ============================================================
# ADVANCED DATA GENERATORS
# ============================================================

print(f"\nPreparing data with aggressive augmentation...")

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # NEW: lighting variations
    shear_range=0.2,  # NEW: elastic deformations
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
print(f"✅ Advanced augmentation enabled")

# ============================================================
# LOAD PHASE 5 CHECKPOINT
# ============================================================

print("\nLoading Phase 5 checkpoint...")
model = tf.keras.models.load_model(PHASE5_CHECKPOINT)

# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================

print("Computing class weights...")

class_weights_arr = compute_class_weight(
    'balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

sqrt_weights = np.sqrt(class_weights_arr)
class_weight_dict = {k: min(v, 1.8) for k, v in dict(enumerate(sqrt_weights)).items()}

# ============================================================
# RECOMPILE (LOWER LR FOR FINE-TUNING)
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.000002),  # Lower for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled (LR=2e-6 for fine-tuning)")

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

print(f"\n🚀 Training Phase 6 ({EPOCHS} epochs, extended training)...\n")

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
total_epochs = len(history.history['val_accuracy'])

print("\n" + "=" * 60)
print("PHASE 6 RESULTS (EXTENDED TRAINING)")
print("=" * 60)
print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Improvement from Phase 5: +{(best_val_acc - 0.72)*100:.2f}%")
print(f"Total epochs: {total_epochs}")
print(f"\n✅ Key Achievements:")
print(f"   - Aggressive augmentation applied")
print(f"   - Extended training to convergence")
print(f"   - Early stopping if no improvement")
print(f"Model saved: {MODEL_SAVE_PATH}")
print("=" * 60)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.axhline(y=0.72, color='r', linestyle='--', label='Phase 5 (72%)')
plt.title('Phase 6: Accuracy (Extended Training)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Phase 6: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"D:\FYP\ml\models\phase6_history.png", dpi=100)
print("\n✅ Training plot saved")

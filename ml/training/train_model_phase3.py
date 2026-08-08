# ============================================================
# EpiTrack: Phase 3 — Full unfreeze, continue from Phase 2 checkpoint
# ============================================================
# Phase 2 hit its epoch ceiling (60) while STILL IMPROVING
# (val_accuracy 0.5519 -> 0.5555 at the last epoch, no plateau).
# This continues training with the entire base unfrozen and a
# learning rate scheduler for finer-grained progress.
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# --- Paths ---
DATA_ROOT = "/kaggle/input/datasets/azemmeer/epitrack/organized"
TRAIN_PATH = f"{DATA_ROOT}/train"
VAL_PATH = f"{DATA_ROOT}/val"

# UPDATE to wherever your Phase 2 checkpoint is attached as input
PHASE2_CHECKPOINT = "/kaggle/input/epitrack-phase2-checkpoint/efficientnet_b4_phase2_best.h5"

OUTPUT_ROOT = "/kaggle/working"
PHASE3_BEST_PATH = f"{OUTPUT_ROOT}/efficientnet_b4_phase3_best.h5"
FINAL_MODEL_PATH = f"{OUTPUT_ROOT}/efficientnet_b4_final.h5"

print("GPU available:", tf.config.list_physical_devices('GPU'))

# --- Data generators (NO rescale) ---
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
    TRAIN_PATH, target_size=(224, 224), batch_size=32, shuffle=True
)
val_data = val_datagen.flow_from_directory(
    VAL_PATH, target_size=(224, 224), batch_size=32, shuffle=False
)

NUM_CLASSES = len(train_data.class_indices)
print(f"Detected {NUM_CLASSES} classes")

# --- Class weights ---
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights_arr))
class_weight_dict = {k: min(v, 3.0) for k, v in class_weight_dict.items()}

# --- Load Phase 2 model, unfreeze everything ---
print("Loading Phase 2 checkpoint...")
model = tf.keras.models.load_model(PHASE2_CHECKPOINT)
print("Loaded.")

for layer in model.layers[0].layers:  # base_model is layers[0] in the Sequential
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.000005),  # very low LR for full fine-tune
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- Callbacks, including LR scheduler ---
checkpoint = ModelCheckpoint(
    PHASE3_BEST_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=6, min_lr=1e-7, verbose=1
)

# --- Train ---
history3 = model.fit(
    train_data, validation_data=val_data, epochs=100,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight_dict, verbose=1
)

# --- Save + plot ---
model.save(FINAL_MODEL_PATH)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history3.history['accuracy'], label='Train')
plt.plot(history3.history['val_accuracy'], label='Validation')
plt.title('Phase 3 Accuracy')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history3.history['loss'], label='Train')
plt.plot(history3.history['val_loss'], label='Validation')
plt.title('Phase 3 Loss')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_ROOT}/training_history_phase3.png")
print("Done. Best model:", PHASE3_BEST_PATH)

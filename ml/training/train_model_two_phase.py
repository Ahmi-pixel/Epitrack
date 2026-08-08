# ============================================================
# EpiTrack: Two-Phase Training (Kaggle)
# ============================================================
# Fixes applied after debugging majority-class collapse:
# - No rescale=1./255 (EfficientNetB4 has its own internal normalization —
#   double-rescaling was destroying the input signal in earlier attempts)
# - Two-phase training: head-only first (higher LR), then fine-tune
#   with more unfrozen layers (low LR)
# - Starting fresh from ImageNet weights
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# --- Paths (adjust for your environment: Kaggle input/working, or local) ---
DATA_ROOT = "/kaggle/input/datasets/azemmeer/epitrack/organized"
TRAIN_PATH = f"{DATA_ROOT}/train"
VAL_PATH = f"{DATA_ROOT}/val"

OUTPUT_ROOT = "/kaggle/working"
PHASE1_BEST_PATH = f"{OUTPUT_ROOT}/efficientnet_b4_phase1_best.h5"
PHASE2_BEST_PATH = f"{OUTPUT_ROOT}/efficientnet_b4_phase2_best.h5"
FINAL_MODEL_PATH = f"{OUTPUT_ROOT}/efficientnet_b4_final.h5"
CLASS_INDICES_PATH = f"{OUTPUT_ROOT}/class_indices.json"

print("GPU available:", tf.config.list_physical_devices('GPU'))

# --- Data generators (NO rescale — EfficientNet normalizes internally) ---
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

with open(CLASS_INDICES_PATH, 'w') as f:
    json.dump(train_data.class_indices, f, indent=2)

# --- Class weights (capped at 3.0 to reduce instability) ---
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights_arr))
class_weight_dict = {k: min(v, 3.0) for k, v in class_weight_dict.items()}
print("Capped class weights:", class_weight_dict)

# --- Build model fresh from ImageNet ---
base_model = EfficientNetB4(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # PHASE 1: fully frozen

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- PHASE 1: train head only ---
print("\n=== PHASE 1: Training dense head only (base frozen) ===")
phase1_checkpoint = ModelCheckpoint(
    PHASE1_BEST_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
)
phase1_early_stop = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
)
history1 = model.fit(
    train_data, validation_data=val_data, epochs=15,
    callbacks=[phase1_checkpoint, phase1_early_stop],
    class_weight=class_weight_dict, verbose=1
)

# --- Sanity check: confirm collapse is broken before Phase 2 ---
val_data.reset()
preds = model.predict(val_data, verbose=1)
predicted_classes = np.argmax(preds, axis=1)
unique, counts = np.unique(predicted_classes, return_counts=True)
idx_to_class = {v: k for k, v in train_data.class_indices.items()}

print("\nPredicted class distribution after Phase 1:")
for idx, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
    print(f"  {idx_to_class[idx]}: {count} ({count/len(predicted_classes)*100:.1f}%)")
print(f"Distinct classes predicted: {len(unique)} / {NUM_CLASSES}")

# --- PHASE 2: unfreeze last 100 layers, fine-tune together ---
print("\n=== PHASE 2: Fine-tuning with last 100 layers unfrozen ===")
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase2_checkpoint = ModelCheckpoint(
    PHASE2_BEST_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
)
phase2_early_stop = EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
)
history2 = model.fit(
    train_data, validation_data=val_data, epochs=60,
    callbacks=[phase2_checkpoint, phase2_early_stop],
    class_weight=class_weight_dict, verbose=1
)

# --- Save + plot ---
model.save(FINAL_MODEL_PATH)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='Train')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='Validation')
plt.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', label='Phase 1->2')
plt.title('Model Accuracy')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'] + history2.history['loss'], label='Train')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='Validation')
plt.axvline(x=len(history1.history['loss']), color='gray', linestyle='--', label='Phase 1->2')
plt.title('Model Loss')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_ROOT}/training_history_two_phase.png")
print("Done. Final model saved to:", FINAL_MODEL_PATH)

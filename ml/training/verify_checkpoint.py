# ============================================================
# EpiTrack: Verify checkpoint accuracy (no training)
# Confirms the downloaded .h5 file's true val_accuracy in isolation,
# with zero risk of accidentally training further.
# ============================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ---- Paths ----
ORGANIZED_DATA = "/kaggle/input/datasets/azemmeer/epitrack/organized"
VAL_PATH = f"{ORGANIZED_DATA}/val"
CHECKPOINT_PATH = "/kaggle/input/<your-dataset>/model.h5"

IMG_SIZE = 380
BATCH_SIZE = 16

# ---- Val data (no augmentation, shuffle=False) ----
val_datagen = ImageDataGenerator()
val_data = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

NUM_CLASSES = len(val_data.class_indices)
print(f"✅ {NUM_CLASSES} classes, {val_data.samples} val samples")

# ---- Rebuild the exact same architecture ----
base_model = EfficientNetB4(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))

# ---- Load the checkpoint ----
print(f"Loading checkpoint: {CHECKPOINT_PATH}")
model.load_weights(CHECKPOINT_PATH)
print("✅ Weights loaded")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---- Evaluate, nothing else ----
val_loss, val_acc = model.evaluate(val_data, verbose=1)

print(f"\n✅ Verified val_accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Verified val_loss: {val_loss:.4f}")

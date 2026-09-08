# ============================================================
# EpiTrack: Confusion Matrix — Phase 5 checkpoint
# Run this AFTER training to analyze model performance
# Produces classification report + confusion matrix heatmap
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Assumes model and val_data already loaded

# Reset generator to be safe
val_data.reset()

print("Running predictions on validation set...")
y_pred_probs = model.predict(val_data, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_data.classes
class_names = list(val_data.class_indices.keys())

# Full classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# Full confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# ---- Zoom in on the texture cluster ----
texture_classes = [
    'Psoriasis pictures Lichen Planus and related diseases',
    'Eczema Photos',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Nail Fungus and other Nail Disease',
    'Scabies Lyme Disease and other Infestations and Bites'
]

present = [c for c in texture_classes if c in class_names]
missing = [c for c in texture_classes if c not in class_names]

if missing:
    print(f"\n⚠️ These class names weren't found as-is: {missing}")

if present:
    texture_cm = cm_df.loc[present, present]
    print("\n" + "="*60)
    print("TEXTURE CLUSTER CONFUSION (subset)")
    print("="*60)
    print(texture_cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(texture_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present, yticklabels=present)
    plt.title('Texture Cluster Confusion')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('/kaggle/working/texture_cluster_confusion.png', dpi=150)
    plt.show()

# ---- Full heatmap for reference ----
plt.figure(figsize=(16, 14))
sns.heatmap(cm_df, annot=False, cmap='Blues')
plt.title('Full Confusion Matrix')
plt.tight_layout()
plt.savefig('/kaggle/working/full_confusion_matrix.png', dpi=150)
plt.show()

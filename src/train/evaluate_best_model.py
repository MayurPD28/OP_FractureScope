import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import SynthDataset
from src.models.early_fusion import EarlyFusionCNN
from src.data.transforms import val_transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set constants
TEST_DIR = "/home/mayur/OP_FractureScope/data/synth/test_dataset"
MODEL_PATH = "/home/mayur/OP_FractureScope/outputs/models/best_acc_model_20250505-060834.pth"
RESULTS_DIR = "outputs/test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CLASS_NAMES = ["cracked", "corrosion", "leak"]

# Load model
print(f" Loading model from: {MODEL_PATH}")
model = EarlyFusionCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss()

# Load test dataset
test_dataset = SynthDataset(root_dir=TEST_DIR, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation loop
all_preds = []
all_labels = []
total_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
avg_loss = total_loss / len(test_dataset)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)

# Save classification report
report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(RESULTS_DIR, "classification_report.csv")
report_df.to_csv(report_csv_path)

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
conf_matrix_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()

# Save results summary
summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
with open(summary_path, 'w') as f:
    f.write("=== Test Evaluation Summary ===\n")
    f.write(f"Total Test Samples: {len(test_dataset)}\n")
    f.write(f"Test Loss: {avg_loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test Precision: {precision:.4f}\n")
    f.write(f"Test Recall: {recall:.4f}\n")
    f.write(f"Test F1 Score: {f1:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(report_df.to_string())

# Final print summary
print("\nEvaluation Complete!")
print(f"Saved confusion matrix: {conf_matrix_path}")
print(f"Saved classification report: {report_csv_path}")
print(f"Saved summary: {summary_path}")

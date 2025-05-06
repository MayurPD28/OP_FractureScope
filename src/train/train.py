import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import SynthDataset
from src.data.transforms import train_transforms, val_transforms
from src.models.early_fusion import EarlyFusionCNN
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directories
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_dir = f"outputs/tensorboard/{timestamp}"
model_dir = f"outputs/models"
metrics_dir = f"outputs/metrics"

os.makedirs(tensorboard_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# TensorBoard Writer
writer = SummaryWriter(log_dir=tensorboard_dir)

# Dataset paths
root_dir = "data/synth"

# Transforms
train_transform = train_transforms
val_transform =  val_transforms

# Datasets and Dataloaders
train_dataset = SynthDataset(root_dir, transform=train_transform, split="train")
val_dataset = SynthDataset(root_dir, transform=val_transform, split="val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = EarlyFusionCNN(num_classes=3).to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Early stopping parameters
patience = 3  # Number of epochs with no improvement before stopping
best_val_loss = float('inf')
epochs_without_improvement = 0

# Variables for tracking loss/accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training Loop
num_epochs = 30
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss, running_corrects = 0.0, 0

    loop = tqdm(train_loader, desc="Training", leave=False)
    for fused_imgs, labels in loop: 
        fused_imgs, labels = fused_imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(fused_imgs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * fused_imgs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    writer.add_scalar("Train/Loss", epoch_loss, epoch)
    writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

    # Store for plot
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    # Validation
    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for fused_imgs, labels in val_loader:
            fused_imgs, labels = fused_imgs.to(device), labels.to(device)

            outputs = model(fused_imgs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * fused_imgs.size(0)
            val_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    # Save best model based on accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, f"best_acc_model_{timestamp}.pth"))
        print(f"Saved new best accuracy model with val_acc = {val_acc:.4f}")

    writer.add_scalar("Val/Loss", val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)

    # Store for plot
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    scheduler.step()

    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    
    # Early stopping based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save best model
        torch.save(model.state_dict(), os.path.join(model_dir, f"early_fusion_best_loss{timestamp}.pth"))
        print(f"Validation loss improved to {val_loss:.4f}, saved model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement in val loss for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement in validation loss for {patience} epochs.")
            break

    
# Save classification report & confusion matrix
report = classification_report(all_labels, all_preds, target_names=["cracked", "corrosion", "leak"])
conf_matrix = confusion_matrix(all_labels, all_preds)

# Save report
with open(os.path.join(metrics_dir, f"classification_report_{timestamp}.txt"), "w") as f:
    f.write(report)

# Plot and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["cracked", "corrosion", "leak"],
            yticklabels=["cracked", "corrosion", "leak"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png"))
plt.close()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
plt.plot(range(len(val_losses)), val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig(os.path.join(metrics_dir, f"loss_plot_{timestamp}.png"))
plt.close()

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_accuracies)), train_accuracies, label="Train Accuracy")
plt.plot(range(len(val_accuracies)), val_accuracies, label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.savefig(os.path.join(metrics_dir, f"accuracy_plot_{timestamp}.png"))
plt.close()

print("\nTraining complete!")
print(f"Best Val Accuracy: {best_val_acc:.4f}")
print(f"Model saved to: {model_dir}")
print(f"Classification report and confusion matrix saved to: {metrics_dir}")

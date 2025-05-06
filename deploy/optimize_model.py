import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import os

# Importing our model
from src.models.early_fusion import EarlyFusionCNN

# Load trained model
checkpoint_path = "/home/mayur/OP_FractureScope/outputs/models/best_acc_model_20250505-060834.pth"
device = torch.device("cpu")

model = EarlyFusionCNN()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Apply Pruning
def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # For clean export
    return model

model = apply_pruning(model, amount=0.3)

# Apply Quantization
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Save Optimized Model
optimized_path = "deploy/optim_models/optimizedEarlyFusionCNN.pth"
os.makedirs(os.path.dirname(optimized_path), exist_ok=True)
torch.save(model.state_dict(), optimized_path)
print(f"Optimized model saved at: {optimized_path}")

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.onnx
import os
from src.models.early_fusion import EarlyFusionCNN

# Paths
checkpoint_path = "outputs/models/best_acc_model_20250505-060834.pth"
onnx_output_path = "deploy/optim_models/onnx/EarlyFusionCNN_pruned.onnx"
os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)

# Load model
model = EarlyFusionCNN()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# Apply pruning only
def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

model = apply_pruning(model, amount=0.3)

# Export to ONNX
dummy_input = torch.randn(1, 6, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12
)

print(f"ONNX model exported (pruned only) to: {onnx_output_path}")

import torch.nn as nn
import timm

class EarlyFusionCNN(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=3, in_chans=6, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)
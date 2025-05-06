import os
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .transforms import train_transforms, val_transforms

LABEL_MAP = {"cracked": 0, "corrosion": 1, "leak": 2}

class SynthDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, val_ratio=0.2, seed=42):
        self.transform = transform
        self.samples = []
        for cls, label in LABEL_MAP.items():
            vis_dir = os.path.join(root_dir, cls, "visual")
            therm_dir = os.path.join(root_dir, cls, "thermal")
            filenames = os.listdir(vis_dir)
            for fname in filenames:
                self.samples.append({
                    "visual": os.path.join(vis_dir, fname),
                    "thermal": os.path.join(therm_dir, fname),
                    "label": label
                })

        # Split once
        train_samples, val_samples = train_test_split(self.samples, test_size=val_ratio, random_state=seed, stratify=[s['label'] for s in self.samples])
        self.samples = train_samples if split == "train" else val_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        vis = cv2.imread(item["visual"])[:, :, ::-1]
        therm = cv2.imread(item["thermal"])[:, :, ::-1]
        if self.transform:
            augmented = self.transform(image=vis, thermal=therm)
            vis = augmented["image"]
            therm = augmented["thermal"]
        # fuse channels
        fused = torch.cat([vis, therm], dim=0)
        return fused, torch.tensor(item["label"], dtype=torch.long)
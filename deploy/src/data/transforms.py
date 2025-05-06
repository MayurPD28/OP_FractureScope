import albumentations as A
from albumentations.pytorch import ToTensorV2

# Shared image size
IMG_SIZE = 224

# Define transforms for a single image (used on both visual & thermal)
def base_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),  # Corrected here
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={'thermal': 'image'})
    else:
        return A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={'thermal': 'image'})


# These will be imported in dataset.py
train_transforms = base_transforms(train=True)
val_transforms = base_transforms(train=False)

# Torch related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Default imports for file handling
import os
from pathlib import Path

# Libraries for images
from PIL import Image

# ML Flow related libraries
import mlflow
import mlflow.pytorch

# Matplotlib Libraries for plotting images
import matplotlib.pyplot as plt


class WaterBodiesDataset(Dataset):
    def __init__(self, image_dir, masks_dir, transform=None):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    # Two definitions related to DataLoader
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Because both of them contains the same image names, using self.images
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])

        # Converting the image to RGB values
        image = Image.open(image_path).convert("RGB")

        # L is called Luminance. It is essentially converting the pixel values between 0, 255
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def main():
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    image_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Images")
    masks_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Masks")


if __name__ == "__main__":
    main()

# Torch related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# For Model visualization
from torchview import draw_graph

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

# Importing class from other pages
from model.UNet import UNet

"""
All these functions inside the class is required by the DataLoader for its operations
"""


class WaterBodiesDataset(Dataset):
    """
    The input is the directory path for the images, directory path for the masks, and by default the transform parameter is None, but you can pass through a transform argument
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    # Return the length of the image directory
    def __len__(self):
        return len(self.images)

    # Get images for that particular idx
    def __getitem__(self, idx):
        # Path for the particular image
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Converting the image to RGB Values
        image = Image.open(image_path).convert("RGB")

        # Getting the luminance values for black and white images
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image), self.transform(mask)

        return image, mask


def main():
    # Use CUDA if available else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resize the image to strict 256, 256 without preserving the aspect ratio
    transform = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )

    # Directories for Mask and Images
    image_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Images")
    mask_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Masks")

    # Creating an instance of the WaterBodiesDataset class
    dataset = WaterBodiesDataset(image_dir, mask_dir, transform)

    # Creating an instance of dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)

    img_value = torch.randn(1, 3, 512, 512)

    output = model(img_value.to(device))

    print(output.shape)


if __name__ == "__main__":
    main()

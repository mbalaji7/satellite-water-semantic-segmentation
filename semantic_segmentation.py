# Torch related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

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

# Importing Metrics and Logging libraries
from lib.metrics import dice_coeff, log_prediction_image

# For showing the progress bar
from tqdm import tqdm

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
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    # Directories for Mask and Images
    image_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Images")
    mask_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Masks")

    # Creating an instance of the WaterBodiesDataset class
    dataset = WaterBodiesDataset(image_dir, mask_dir, transform)

    VAL_PCT = 0.2
    training_size = int(len(dataset) - len(dataset) * VAL_PCT)
    testing_size = len(dataset) - training_size

    training_dataset, test_dataset = random_split(
        dataset, [training_size, testing_size]
    )

    # Creating an instance of dataloader
    train_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)

    # Loss calculations
    loss_function = nn.BCEWithLogitsLoss()

    # Initializing Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Going through the images twenty times
    EPOCHS = 20

    # Starting ML Flow
    mlflow.start_run()

    for epoch in range(EPOCHS):
        model.train()

        # The loss and dice for each epoch
        epoch_loss = 0
        epoch_dice = 0

        for batch_idx, (images, masks) in enumerate(
            tqdm(train_dataloader, desc=f"Processing Epoch {epoch + 1}")
        ):
            model.zero_grad()

            images = images.to(device)
            masks = masks.to(device).squeeze().float()

            output = model(images)
            loss = loss_function(output.squeeze(), masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_coeff(output, masks)

        avg_loss = epoch_loss / len(train_dataloader)
        avg_dice = epoch_dice / len(train_dataloader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.2f} | Dice: {avg_dice:.2f}")

        mlflow.log_metric("Average Training Loss", avg_loss, step=epoch)
        mlflow.log_metric("Average Dice Score", avg_dice, step=epoch)

    mlflow.pytorch.log_model(model, "unet_model")
    mlflow.end_run()

    with torch.no_grad():
        # Go to Evaluation Mode
        model.eval()

        for batch_idx, (images, masks) in enumerate(
            tqdm(test_dataloader, desc=f"Testing Batch {batch_idx + 1}")
        ):
            images = images.to(device)
            masks = masks.to(device).squeeze().float()

            output = model(images)

            loss = loss_function(output.squeeze(), masks)
            dice_score = dice_coeff(output.squeeze(), masks)

            for idx in range(len(images)):
                log_prediction_image(
                    images[idx].cpu(),
                    masks[idx].cpu(),
                    output[idx].cpu(),
                    batch_idx,
                    idx,
                    mlflow,
                )

            mlflow.log_metric("Validation Loss", loss, step=batch_idx)
            mlflow.log_metric("Dice Co-efficient", dice_score, step=batch_idx)


if __name__ == "__main__":
    main()

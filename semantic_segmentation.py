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


class UNetConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Creating the down layers
        self.down1 = UNetConv(3, 64)
        self.down2 = UNetConv(64, 128)
        self.down3 = UNetConv(128, 256)
        self.down4 = UNetConv(256, 512)

        # Creating the Up Layers. Cascading to the down layers
        self.up4 = UNetConv(512 + 256, 256)
        self.up3 = UNetConv(256 + 128, 128)
        self.up2 = UNetConv(128 + 64, 64)

        # Final Layer
        self.final_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Going through the first conv layer
        down_1 = self.down1(x)
        down_1 = F.max_pool2d(down_1, (2, 2))

        # Second down layer
        down_2 = self.down2(down_1)
        down_2 = F.max_pool2d(down_2, (2, 2))

        # Third down layer
        down_3 = self.down3(down_2)
        down_3 = F.max_pool2d(down_3, (2, 2))

        # Fourth final down layer
        down_4 = self.down4(down_3)
        down_4 = F.max_pool2d(down_4, (2, 2))

        """
            Starting with the up layer
            You get the data from the previous layer
            Do an interpolation operation on the previous layer and assign it to the current up layer
            Do a torch.cat with the current up layer and previous down layer
            And do a up
        """
        up_4 = F.interpolate(
            down_4, scale_factor=2, mode="bilinear", align_corners=True
        )
        up_4_concat = torch.cat([up_4, down_3], dim=1)
        up_4 = self.up4(up_4_concat)

        # Going to the next up layer
        up_3 = F.interpolate(up_4, scale_factor=2, mode="bilinear", align_corners=True)
        up_3_concat = torch.cat([up_3, down_2], dim=1)
        up_3 = self.up3(up_3_concat)

        # Going to the next up layer
        up_2 = F.interpolate(up_3, scale_factor=2, mode="bilinear", align_corners=True)
        up_2_concat = torch.cat([up_2, down_1], dim=1)
        up_2 = self.up2(up_2_concat)

        # Final Layer
        up_1 = self.final_layer(up_2)

        return up_1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    image_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Images")
    masks_dir = os.path.join(Path().absolute(), "water_bodies_dataset", "Masks")

    # Creating an instance of the dataset that returns object for the dataloader
    dataset = WaterBodiesDataset(image_dir, masks_dir, transform)

    # Dataloader object
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initializing the network
    net = UNet().to(device)

    x = torch.randn(1, 3, 2000, 2000)

    y = net(x.to(device))

    print(y.shape)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoding convolution layers
        # Down Layer 1
        self.down_11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.down_12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Layer 2
        self.down_21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.down_22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Layer 3
        self.down_31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.down_32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Layer 4
        self.down_41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.down_42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Final Handoff Layer
        self.down_51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.down_52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoding Layers
        # Up Layer 1
        self.up_1t = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.up_12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Up Layer 2
        self.up_2t = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up_22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Up Layer 3
        self.up_3t = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up_32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Up Layer 4
        self.up_4t = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up_42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoding Layers
        # Down Layer 1
        d_11 = F.relu(self.down_11(x))
        d_12 = F.relu(self.down_12(d_11))
        d_1 = self.max_pool1(d_12)

        # Down Layer 2
        d_21 = F.relu(self.down_21(d_1))
        d_22 = F.relu(self.down_22(d_21))
        d_2 = self.max_pool2(d_22)

        # Down Layer 3
        d_31 = F.relu(self.down_31(d_2))
        d_32 = F.relu(self.down_32(d_31))
        d_3 = self.max_pool3(d_32)

        # Down Layer 4
        d_41 = F.relu(self.down_41(d_3))
        d_42 = F.relu(self.down_42(d_41))
        d_4 = self.max_pool4(d_42)

        # Down Layer 5 - Final Handoff Layer
        d_51 = F.relu(self.down_51(d_4))
        d_52 = F.relu(self.down_52(d_51))

        # Encoding Layers
        # Up Layer 1
        u_1t = self.up_1t(d_52)
        u_1t_42 = torch.cat([u_1t, d_42], dim=1)
        u_11 = F.relu(self.up_11(u_1t_42))
        u_12 = F.relu(self.up_12(u_11))

        # Up Layer 2
        u_2t = self.up_2t(u_12)

        u_2t_32 = torch.cat([u_2t, d_32], dim=1)
        u_21 = F.relu(self.up_21(u_2t_32))
        u_22 = F.relu(self.up_22(u_21))

        # Up Layer 3
        u_3t = self.up_3t(u_22)
        u_3t_22 = torch.cat([u_3t, d_22], dim=1)
        u_31 = F.relu(self.up_31(u_3t_22))
        u_32 = F.relu(self.up_32(u_31))

        # Up Layer 4
        u_4t = self.up_4t(u_32)
        u_4t_12 = torch.cat([u_4t, d_12], dim=1)
        u_41 = F.relu(self.up_41(u_4t_12))
        u_42 = F.relu(self.up_42(u_41))

        # Final Output Layer
        output = self.output_layer(u_42)

        return output

import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return decoded






import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, image):
        output = self.conv(image)
        return output

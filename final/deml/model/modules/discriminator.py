import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 100)
        )

    def forward(self, concept):
        classes = self.fc(concept)
        return classes

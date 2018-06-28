import torch.nn as nn


class MetaLearning(nn.Module):
    def __init__(self):
        super(MetaLearning, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(1024, 5)
        )

    def forward(self, concept):
        output = self.fc(concept)
        return output

import torch.nn as nn


class classfier(nn.Module):
    def __init__(self, feature_size=2048):
        super(classfier, self).__init__()
        self.feature_size = feature_size

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 11),
            nn.Softmax(dim=1)
        )

    def forward(self, feature):
        feature = feature.contiguous()  # check
        feature = feature.view(-1, self.feature_size)
        category = self.fc(feature)
        return category

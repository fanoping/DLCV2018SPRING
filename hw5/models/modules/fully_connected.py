import torch.nn as nn


class Classfier(nn.Module):
    def __init__(self, feature_size=4096):
        super(Classfier, self).__init__()
        self.feature_size = feature_size

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 11),
            nn.Softmax(dim=1)
        )

    def forward(self, feature):
        feature = feature.contiguous()  # check
        feature = feature.view(-1, self.feature_size)
        category = self.fc(feature)
        return category

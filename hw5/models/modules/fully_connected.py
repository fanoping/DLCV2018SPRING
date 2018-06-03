import torch.nn as nn


class Classfier(nn.Module):
    def __init__(self, feature_size, mode='cnn'):
        super(Classfier, self).__init__()
        self.feature_size = feature_size

        if mode == 'cnn':
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 11),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size, 11),
            )
        self.output = nn.Softmax(dim=1)

    def forward(self, feature):
        category = self.fc(feature)
        category = self.output(category)
        return category

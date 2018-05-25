from .modules import Resnet50, classfier
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = Resnet50()
        self.fc = classfier()

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, frame):
        cnn_feature = self.cnn(frame)
        category = self.fc(cnn_feature.squeeze())
        return category


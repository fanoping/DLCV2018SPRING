import torch.nn as nn


class ProtoLoss(nn.Module):
    def __init__(self):
        super(ProtoLoss, self).__init__()

    def __protoloss(self, input, target):
        pass

    def forward(self, input, target):
        return self.__protoloss(input, target)

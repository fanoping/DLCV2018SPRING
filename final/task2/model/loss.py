import torch.nn as nn
# TODO: loss function implementation


class ProtoLoss(nn.Module):
    def __init__(self):
        super(ProtoLoss, self).__init__()

    def __protoloss(self, input, target):
        pass

    def forward(self, input, target):
        return 0
        #return self.__protoloss(input, target)

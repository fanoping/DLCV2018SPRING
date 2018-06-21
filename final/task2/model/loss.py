import torch.nn as nn
import torch
# TODO: loss function implementation


class ProtoLoss(nn.Module):
    def __init__(self):
        super(ProtoLoss, self).__init__()

    def forward(self, support_image, support_label, query_image, query_label):
        return self.__protoloss(support_image, support_label, 5)

    def __protoloss(self, input, target, n_support):
        pass

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        code from: https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/utils.py
        '''
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

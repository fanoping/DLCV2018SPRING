from .modules import Encoder, classfier
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.encoder = Encoder()
        self.classifier = classfier(feature_size=4*512)

    def forward(self, frames):
        enc_out, hidden_out = self.encoder(frames)
        category = self.classifier(enc_out)
        return category

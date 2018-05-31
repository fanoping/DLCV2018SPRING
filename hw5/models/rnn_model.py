from .modules import Classfier, Encoder
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.encoder = Encoder(2048)
        self.fc = Classfier(feature_size=512, mode='rnn')

    def forward(self, frames, length):
        """
            frames:
                shape: seq_len x batch size x feature size
        """
        frames = pack_padded_sequence(frames, length)
        enc_out, (hidden_out, c0) = self.encoder(frames)
        category = self.fc(hidden_out[-1])
        return category

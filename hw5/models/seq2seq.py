from .modules import Classfier, Encoder
import torch.nn as nn
import torch


class SEQ2SEQ(nn.Module):
    def __init__(self, args):
        super(SEQ2SEQ, self).__init__()
        self.args = args
        self.encoder = Encoder(input_size=2048, hidden_size=512)
        self.fc = Classfier(feature_size=512, mode='rnn')

    def forward(self, frames, length):
        """
            frames:
                shape: seq_len x batch size x feature size
            output:
                category: seq_len x batch size x category (11)
                mask: seq_len x batch size
        """
        out_seq = []

        enc_out, _ = self.encoder(frames)
        """
        enc_out, enc_out_len = pad_packed_sequence(enc_out)

        
        # mask for calculating loss
        mask = torch.zeros((enc_out.size(0), enc_out.size(1)))
        for idx, single_len in enumerate(length):
            mask[:single_len, idx] = 1
        """
        for idx in range(enc_out.size(0)):
            category = self.fc(enc_out[idx])
            out_seq.append(category)

        category = torch.stack(out_seq)
        return category

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
        )

    def forward(self, in_seq):
        output, hidden = self.rnn(in_seq)
        return output, hidden


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    a = torch.randn(4, 4096).unsqueeze(0)
    a = torch.cat((a, a), dim=0)
    a = Variable(a).cuda()

    model = Encoder().cuda()
    output, hidden = model(a)
    print(a.size())  # batch x 4 x 4096
    print(hidden[0].size(), hidden[1].size())  # num_layers x batch x hidden_size
    print(output.size())  # batch x 4 x 4096

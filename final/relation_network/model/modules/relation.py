import torch.nn as nn
import torch
import math


class Relation(nn.Module):
    def __init__(self, feature_size, hidden_size, relation_dim):
        super(Relation, self).__init__()
        assert(len(hidden_size) == 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_size*2, hidden_size[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size[0], momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size[1], momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size[1], relation_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())

    def forward(self, feature):
        out = self.conv1(feature)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    test = Relation(64, 64)
    import torch
    from torch.autograd import Variable
    a = Variable(torch.ones((1, 128, 6, 6)))

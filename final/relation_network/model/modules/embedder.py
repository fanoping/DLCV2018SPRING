import torch.nn as nn
import torch
import math


class Embedder(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Embedder, self).__init__()
        assert(len(hidden_size) == 3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size[0], kernel_size=3, padding=0),
            nn.BatchNorm2d(hidden_size[0], momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=3, padding=0),
            nn.BatchNorm2d(hidden_size[1], momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size[2], momentum=1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size[2], feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size, momentum=1, affine=True),
            nn.ReLU(inplace=True)
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

    def forward(self, image):
        output = self.conv1(image)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output


if __name__ == '__main__':
    test = Embedder(64, 64)
    import torch
    from torch.autograd import Variable
    a = Variable(torch.ones((1, 3, 28, 28)))

    out = test(a)

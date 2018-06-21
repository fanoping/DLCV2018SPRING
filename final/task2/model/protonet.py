import torch.nn as nn


class Protonet(nn.Module):
    def __init__(self, config):
        super(Protonet, self).__init__()
        hidden_size = config['model']['hidden_size']
        feature_size = config['model']['feature_size']

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # hidden_size x 32 x 32
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # hidden_size x 16 x 16
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # hidden_size x 8 x 8
            nn.Conv2d(hidden_size, feature_size, kernel_size=3, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # feature_size x 1 x 1
        )

    def forward(self, image):
        """
            input
                image: batch size x 3 x 32 x 32
            output
                output: batch size x 64
        """
        output = self.encoder(image)
        return output.view(output.size(0), -1)


if __name__ == '__main__':
    test = Protonet({})
    import torch
    from torch.autograd import Variable
    a = Variable(torch.ones(1, 3, 32, 32))
    k = test(a)
    print(k)

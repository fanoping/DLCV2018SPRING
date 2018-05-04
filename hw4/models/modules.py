import torch
import torch.nn as nn
from torch.autograd import Variable

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv2_2 = nn.Conv2d(8, 16, kernel_size=3)

        self.convt1 = nn.ConvTranspose2d(16, 8, kernel_size=3)
        self.convt2 = nn.ConvTranspose2d(8, 3, kernel_size=3)

    def encoder(self, x):
        hidden = nn.ReLU(self.conv1(x))
        return self.conv2_1(hidden), self.conv2_2(hidden)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        hidden = nn.ReLU(self.convt1(z))
        return nn.ReLU(self.convt2(hidden))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, lm=0.05):
        super(CustomLoss, self).__init__()
        self.lm = lm

    def forward(self, recon_img, img, mu, logvar):
        """
        shape
        recon_img: batch size x image shape (3 x 64 x 64)
        img: batch size x image shape (3 x 64 x 64)
        lamb: lambda, importance of kl divergence
        mu: latent mean
        logvar: latent log variance

        reconstruction loss: MSE loss
        kl divergence: -.5 * (1 + log(sigma^2) - mean^2 - sigma^2)
        """
        loss_fn = nn.MSELoss(size_average=False)
        reconstruction_loss = loss_fn(recon_img, img)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp_())

        return reconstruction_loss + self.lm*kl_divergence

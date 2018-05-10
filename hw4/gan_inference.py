import torch
from models.modules import Discriminator, Generator, ACGANDiscriminator, ACGANGenerator
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import readfigs
import torchvision
from scipy.misc import imsave

if __name__ == '__main__':
    checkpoint = torch.load('checkpoints_acgan/epoch80_checkpoint.pth.tar')
    g_model = ACGANGenerator().cuda()
    d_model = ACGANDiscriminator().cuda()
    g_model.load_state_dict(checkpoint['state_dict'][0])
    d_model.load_state_dict(checkpoint['state_dict'][1])
    g_model.eval()
    d_model.eval()

    torch.manual_seed(1)
    noise = (torch.randn(32, 100))
    fixed_fake_classes = np.zeros((32, 13))
    fixed_fake_classes[:, 9] = 1 # Smiling
    fake_classes = torch.FloatTensor(fixed_fake_classes)
    noise_1 = torch.cat((noise, fake_classes), dim=1)
    noise_1 = Variable(noise_1).cuda()

    fixed_fake_classes = np.zeros((32, 13))
    fake_classes = torch.FloatTensor(fixed_fake_classes)
    noise_2 = torch.cat((noise, fake_classes), dim=1)
    noise_2 = Variable(noise_2).cuda()

    fake_image = g_model(noise_1)
    print(fake_image)
    fake_image = fake_image.mul(0.5).add_(0.5)
    print(fake_image)
    print(fake_image.data[0].size())
    torchvision.utils.save_image(fake_image.data, "saved/acgan_smile_test.png")
    print("Saved")

    fake_image = g_model(noise_2)
    print(fake_image)
    fake_image = fake_image.mul(0.5).add_(0.5)
    print(fake_image)
    print(fake_image.data[0].size())
    torchvision.utils.save_image(fake_image.data, "saved/acgan_no_smile_test.png")
    print("Saved")

import torch
import torchvision
from torch.autograd import Variable
from models.modules import INFOGANDiscriminator, INFOGANGenerator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def main(args):

    output_file = os.path.join(args.output_file)
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    if not os.path.exists(args.checkpoint):
        print("{} not exists".format(args.checkpoint))
        return

    # gpu configuration
    with_cuda = not args.no_cuda

    checkpoint = torch.load(args.checkpoint)
    g_model = INFOGANGenerator().cuda() if with_cuda else INFOGANGenerator()
    d_model = INFOGANDiscriminator().cuda() if with_cuda else INFOGANDiscriminator()
    g_model.load_state_dict(checkpoint['state_dict'][0])
    d_model.load_state_dict(checkpoint['state_dict'][1])

    g_model.eval()
    d_model.eval()

    # 4-3
    print("Random Generation of 10(Discrete) images......")
    fix = np.zeros((10, 100))
    for i in range(10):
        for j in range(10):
            fix[i, 10*j + i] = 1.0

    discrete = torch.FloatTensor(fix)
    discrete_noise = Variable(discrete)  # 10 * 100

    fixed_noise = torch.FloatTensor()  # 10 * 128
    for _ in range(10):
        torch.manual_seed(388)
        tmp = torch.randn(1, 128)
        fixed_noise = torch.cat((fixed_noise, tmp), dim=0)

    discrete_noise = discrete_noise.cuda() if with_cuda else discrete_noise
    fixed_noise = Variable(fixed_noise).cuda() if with_cuda else Variable(fixed_noise)

    noise = torch.cat((fixed_noise, discrete_noise), dim=1)

    result = g_model(noise)
    result = result.mul(0.5).add_(0.5)

    filename = os.path.join(output_file, 'fig4_3.jpg')
    torchvision.utils.save_image(result.data, filename, nrow=10)

    # 4-2
    print("Saving loss figure......")
    real_loss_list = checkpoint['loss']['real_loss']
    fake_loss_list = checkpoint['loss']['fake_loss']
    x_label = [i for i in range(1, len(real_loss_list) + 1)]

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Real/Fake Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x_label, real_loss_list, 'b', label='real loss')
    plt.plot(x_label, fake_loss_list, 'r', label='fake loss')
    plt.legend(loc="best")

    print("Saving accuracy figure......")
    real_acc_list = checkpoint['accuracy']['real_acc']
    fake_acc_list = checkpoint['accuracy']['fake_acc']

    plt.subplot(122)
    plt.title('Real/Fake Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(x_label, real_acc_list, 'b', label='real accuracy')
    plt.plot(x_label, fake_acc_list, 'r', label='fake accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    filename = os.path.join(output_file, 'fig4_2.jpg')
    plt.savefig(filename)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="INFOGAN inference")
    parser.add_argument('--output-file', default='saved/infogan',
                        help='output data directory')
    parser.add_argument('--checkpoint',
                        default='checkpoints/infogan/epoch86_checkpoint.pth.tar',
                        help='load checkpoint')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())

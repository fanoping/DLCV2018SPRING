import torch
import torchvision
from torch.autograd import Variable
from models.modules import ACGANDiscriminator, ACGANGenerator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def main(args):
    torch.manual_seed(1)

    output_file = os.path.join(args.output_file)
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    if not os.path.exists(args.checkpoint):
        return "{} not exists".format(args.checkpoint)

    # gpu configuration
    with_cuda = not args.no_cuda

    checkpoint = torch.load(args.checkpoint)
    g_model = ACGANGenerator().cuda() if with_cuda else ACGANDiscriminator()
    d_model = ACGANDiscriminator().cuda() if with_cuda else ACGANDiscriminator()
    g_model.load_state_dict(checkpoint['state_dict'][0])
    d_model.load_state_dict(checkpoint['state_dict'][1])

    g_model.eval()
    d_model.eval()

    # 3-3
    print("Random Generation of 10 images......")
    noise = torch.randn(10, 100).view(-1, 100, 1, 1)

    # random classify
    classes_no_smile = np.random.randint(0, 1, size=(10, 13))
    classes_with_smile = np.copy(classes_no_smile)

    # for no smiling
    classes_no_smile[:, 9] = 0
    classes_no_smile = torch.FloatTensor(classes_no_smile)

    # for smiling
    classes_with_smile[:, 9] = 1
    classes_with_smile = torch.FloatTensor(classes_with_smile)

    noise1 = torch.cat((noise, classes_no_smile), dim=1)
    noise1 = Variable(noise1).cuda() if with_cuda else Variable(noise1)
    fake_image1 = g_model(noise1)
    fake_image1 = fake_image1.mul(0.5).add_(0.5)

    noise2 = torch.cat((noise, classes_with_smile), dim=1)
    noise2 = Variable(noise2).cuda() if with_cuda else Variable(noise2)
    fake_image2 = g_model(noise2)
    fake_image2 = fake_image2.mul(0.5).add_(0.5)

    result = torch.cat((fake_image1.data, fake_image2.data), dim=0)

    filename = os.path.join(output_file, 'fig3_3.jpg')
    torchvision.utils.save_image(result.data, filename, nrow=10)

    # 3-2
    print("Saving loss figure......")
    real_loss_list = checkpoint['loss']['real_loss']
    fake_loss_list = checkpoint['loss']['fake_loss']
    x_label = [i for i in range(1, len(real_loss_list)+1)]

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
    plt.ylabel('loss')
    plt.plot(x_label, real_acc_list, 'b', label='real accuracy')
    plt.plot(x_label, fake_acc_list, 'r', label='fake accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    filename = os.path.join(output_file, 'fig3_2.jpg')
    plt.savefig(filename)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GAN inference")
    parser.add_argument('--output-file', default='saved/gan',
                        help='output data directory')
    parser.add_argument('--checkpoint',
                        default='checkpoints/gan/epoch10_checkpoint.pth.tar',
                        help='load checkpoint')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())


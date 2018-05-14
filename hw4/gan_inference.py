import torch
import torchvision
from torch.autograd import Variable
from models.modules import GANDiscriminator, GANGenerator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import os


def main(args):
    torch.manual_seed(1337)

    output_file = os.path.join(args.output_file)
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    if not os.path.exists(args.checkpoint):
        print("{} not exists".format(args.checkpoint))
        return

    # gpu configuration
    with_cuda = not args.no_cuda

    checkpoint = torch.load(args.checkpoint)
    g_model = GANGenerator().cuda() if with_cuda else GANDiscriminator()
    d_model = GANDiscriminator().cuda() if with_cuda else GANDiscriminator()
    g_model.load_state_dict(checkpoint['state_dict'][0])
    d_model.load_state_dict(checkpoint['state_dict'][1])

    g_model.eval()
    d_model.eval()

    # 2-3
    print("Random Generation......")
    noise = (torch.randn(32, 100)).view(-1, 100, 1, 1)
    noise = Variable(noise).cuda() if with_cuda else Variable(noise)
    fake_image = g_model(noise)
    fake_image = fake_image.mul(0.5).add_(0.5)
    filename = os.path.join(output_file, 'fig2_3.jpg')
    torchvision.utils.save_image(fake_image.data, filename)

    # 2-2
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
    plt.ylabel('accuracy')
    plt.plot(x_label, real_acc_list, 'b', label='real accuracy')
    plt.plot(x_label, fake_acc_list, 'r', label='fake accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    filename = os.path.join(output_file, 'fig2_2.jpg')
    plt.savefig(filename)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GAN inference")
    parser.add_argument('--output-file', default='saved/gan',
                        help='output data directory')
    parser.add_argument('--checkpoint',
                        default='checkpoints/gan/epoch231_checkpoint.pth.tar',
                        help='load checkpoint')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())


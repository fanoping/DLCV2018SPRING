import torch
import torchvision
from torchvision import transforms
from models.modules import VAE
from torch.autograd import Variable
from utils.preprocess import readfigs
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def main(args):
    torch.manual_seed(1337)

    if not os.path.exists(args.checkpoint):
        return "{} not exists".format(args.checkpoint)

    # gpu configuration
    with_cuda = not args.no_cuda

    checkpoint = torch.load(args.checkpoint)
    model = VAE().cuda() if with_cuda else VAE()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    y_test = readfigs('hw4_data/test')
    print(y_test.shape)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test = torch.FloatTensor().cuda() if with_cuda else torch.FloatTensor()
    result = torch.FloatTensor().cuda() if with_cuda else torch.FloatTensor()

    """
    print("Predicting......")
    for i in range(len(y_test)):
        test_img = transform(y_test[i])
        test_img = Variable(test_img).cuda() if with_cuda else Variable(test_img)
        test_x = test_img.mul(0.5).add_(0.5)
        test = torch.cat((test, test_x.unsqueeze(0).data), dim=0)
        y_pred, _, _ = model(test_img.unsqueeze(0))
        y_pred = y_pred.mul(0.5).add_(0.5)
        result = torch.cat((result, y_pred.data), dim=0)

    # Save 10 images
    print("Saving VAE reconstruction of 10 images......")
    result = torch.cat((test[:10], result[:10]), dim=0)
    torchvision.utils.save_image(result, "saved/vae/test.png", nrow=10)
    

    print("Saving loss figure......")
    mse_loss_list = checkpoint['mse_loss']
    kld_loss_list = checkpoint['kld_loss']
    x_label = [i for i in range(1, len(mse_loss_list)+1)]

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Reconstruction (MSE) Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x_label, mse_loss_list, 'r', label='mse loss')
    plt.legend(loc="best")

    plt.subplot(122)
    plt.plot(x_label, kld_loss_list, 'b', label='kld loss')
    plt.title('KL Divergence Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('saved/vae/loss.png')

    """


    print("Saving random generation......")
    decoder = model.decoder
    noise = torch.randn(32, 1024).view(-1, 1024, 1, 1)
    noise = Variable(noise).cuda() if with_cuda else Variable(noise)
    predict = decoder(noise)
    predict = predict.mul(0.5).add_(0.5)
    torchvision.utils.save_image(predict.data, "saved/vae/noise.png", nrow=8)

    print("Done")


if __name__ == '__main__':
    """
    func_name_list = ['sinc', 'stair']
    func_lambda_list = [lambda x: np.sin(4 * np.pi * x) / (4 * np.pi * x + 1e-10),
                        lambda x: (np.ceil(4 * x) - 2.5) / 1.5]
    arch_list = ['Deep', 'Middle', 'Shallow']
    base_arch = 'FC'
    color_list = ['r', 'g', 'b']

    plt.figure(figsize=(12, 9))
    for i, func_name in enumerate(func_name_list):
        func = func_lambda_list[i]
        x = np.array([i for i in np.linspace(0, 1, 10000)])
        y_target = np.array([func(i) for i in x])
        plt.title(func_name+' loss')
        plt.subplot(220 + i + 1)
        plt.plot(x, y_target, 'k', label='Ground truth')
        for arch, color in zip(arch_list, color_list):
            checkpoint = torch.load('../../models/saved/1-1/'+arch+base_arch+'_'+func_name+'_checkpoint.pth.tar')
            model = eval(checkpoint['arch'])()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            y_pred = np.array([model(Variable(torch.FloatTensor(np.array([[i]])))).data.numpy() for i in x]).squeeze()
            plt.plot(x, y_pred, color, label=arch+base_arch)
            plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
    """
    parser = argparse.ArgumentParser(description="VAE inference")
    parser.add_argument('--checkpoint', required=False,
                        default='checkpoints/vae/best_checkpoint.pth.tar',
                        help='load checkpoint')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())

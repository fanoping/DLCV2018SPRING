import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from models.modules import VAE
from torch.autograd import Variable
from utils.preprocess import readfigs
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
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
    model = VAE('test').cuda() if with_cuda else VAE('test')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    y_test = readfigs(args.input_file)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 1-3
    test = torch.FloatTensor().cuda() if with_cuda else torch.FloatTensor()
    result = torch.FloatTensor().cuda() if with_cuda else torch.FloatTensor()

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
    filename = os.path.join(output_file, 'fig1_3.jpg')
    torchvision.utils.save_image(result, filename, nrow=10)
    print("Computing MSE Loss:", end=' ')
    loss_fn = nn.MSELoss()
    origin = Variable(result[:10]).cuda() if with_cuda else Variable(result[:10])
    result = Variable(result[10:]).cuda() if with_cuda else Variable(result[10:])
    recon_loss = loss_fn(origin, result)
    print(recon_loss.data[0])

    # 1-2
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

    filename = os.path.join(output_file, 'fig1_2.jpg')
    plt.savefig(filename)

    # 1-4
    print("Saving random generation......")
    decoder = model.decoder
    noise = torch.randn(32, 1024).view(-1, 1024, 1, 1)
    noise = Variable(noise).cuda() if with_cuda else Variable(noise)
    predict = decoder(noise)
    predict = predict.mul(0.5).add_(0.5)

    filename = os.path.join(output_file, 'fig1_4.jpg')
    torchvision.utils.save_image(predict.data, filename, nrow=8)

    # 1-5
    print('Computing TSNE......')

    tsne = TSNE(n_components=2, random_state=20, verbose=1, n_iter=1000)
    encoder = model.encoder
    mu = model.mu

    latent = torch.FloatTensor().cuda() if with_cuda else torch.FloatTensor()

    print("Encoding......")
    for i in range(len(y_test)):
        test_img = transform(y_test[i])
        test_img = Variable(test_img).cuda() if with_cuda else Variable(test_img)
        encoded = encoder(test_img.unsqueeze(0))
        latent_output = mu(encoded)
        latent = torch.cat((latent, latent_output.data), dim=0)

    latent_sample = latent.squeeze().cpu().numpy()
    embedded_latent = tsne.fit_transform(latent_sample)

    gender = pd.read_csv('hw4_data/test.csv')
    gender = gender.ix[:, 8].as_matrix().astype('float')

    plt.figure(figsize=(6, 6))
    plt.title('tSNE results')
    i, j = 0, 0
    for lat, gen in zip(embedded_latent, gender):
        if gen == 1.0:
            plt.scatter(lat[0], lat[1], c='b', alpha=0.3, label='male' if  i == 0 else '')
            i += 1
        else:
            plt.scatter(lat[0], lat[1], c='r', alpha=0.3, label='female' if  j == 0 else '')
            j += 1

    plt.legend(loc="best")
    filename = os.path.join(output_file, 'fig1_5.jpg')
    plt.savefig(filename)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAE inference")
    parser.add_argument('--input-file', default='hw4_data/test',
                        help='input data directory')
    parser.add_argument('--output-file', default='saved/vae',
                        help='output data directory')
    parser.add_argument('--checkpoint',
                        default='checkpoints/vae/best_checkpoint.pth.tar',
                        help='load checkpoint')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())

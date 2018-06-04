import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.seq2seq import SEQ2SEQ
from utils.dataset import FullLengthVideo
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    model = SEQ2SEQ(args).cuda() if with_cuda else SEQ2SEQ(args)
    model.load_state_dict(checkpoint['state_dict'])

    # dataset
    valid_dataset = FullLengthVideo(args, 'eval')
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                   batch_size=1,
                                   collate_fn=valid_dataset.rnn_collate_fn,
                                   shuffle=False)

    # 3-3
    result = []
    # ground_truth = []
    # accuracy = 0

    with torch.no_grad():
        print("Predicting......")
        model.eval()

        for video, _, length in valid_data_loader:
            video = Variable(video).cuda() if with_cuda else Variable(video)
            output = model(video, length)
            value, index = torch.max(output, dim=2)
            result.append(index.squeeze())
            # ground_truth.append(label.squeeze())
            # accuracy += np.mean((index.cpu().data == label).numpy())
            # print(np.mean((index.cpu().data == label).numpy()))

    # print("Model average accuracy:", accuracy / len(valid_data_loader))
    files = sorted([file for file in os.listdir(args.full_length_dir) if file.startswith('OP')])

    # ax = plt.subplot(211)
    # ax2 = plt.subplot(212)
    for videos in range(len(result)):
        # write output file
        test = result[videos].cpu().data.numpy().tolist()
        filename = os.path.join(output_file, files[videos])
        with open(filename + '.txt', 'w') as f:
            writeout = [str(test[idx]) + '\n' for idx in range(len(test))]
            f.writelines(writeout)

        """
        colors = ['w', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'C0', 'C1', 'C2']

        cmap = mpl.colors.ListedColormap([colors[idx] for idx in test])
        bounds = [i for i in range(len(test))]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        spacing='proportional',
                                        orientation='horizontal')
        ax.set_ylabel('Predict')

        test = ground_truth[videos].data.numpy().tolist()
        cmap = mpl.colors.ListedColormap([colors[idx] for idx in test])
        bounds = [i for i in range(len(test))]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        spacing='proportional',
                                        orientation='horizontal')
        ax2.set_ylabel('Ground Truth')

        filename = os.path.join(output_file, '{}_seq.jpg'.format(files[videos][:-4]))
        plt.savefig(filename)
        """

    # 3-2
    print("Saving loss figure......")
    loss_list = checkpoint['loss']
    val_loss_list = checkpoint['val_loss']

    x_label = [i for i in range(1, len(loss_list)+1)]

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Training/Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x_label, loss_list, 'b', label='train loss')
    plt.plot(x_label, val_loss_list, 'r', label='valid loss')
    plt.legend(loc="best")

    print("Saving accuracy figure......")
    acc_list = checkpoint['accuracy']
    val_acc_list = checkpoint['val_accuracy']

    plt.subplot(122)
    plt.title('Training/Validation Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(x_label, acc_list, 'b', label='train accuracy')
    plt.plot(x_label, val_acc_list, 'r', label='valid accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    filename = os.path.join(output_file, 'fig3_2.jpg')
    plt.savefig(filename)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Seq2Seq inference")
    parser.add_argument('--input-feature', default='rnn_full_length_valid_feature.tar',
                        help='input feature file')
    parser.add_argument('--full-length-dir', default='HW5_data/FullLengthVideos/videos/valid',
                        help='video data for train/validation')
    parser.add_argument('--full-length-file', default='valid_full_length_video.tar',
                        help='full length video file for dumping tar')
    parser.add_argument('--output-file', default='saved/seq2seq',
                        help='output data directory')
    parser.add_argument('--checkpoint', default='checkpoints/seq2seq_resnet50/best_checkpoint.pth.tar',
                        help='load checkpoint')
    parser.add_argument('--pretrained', default='Resnet50', type=str,
                        help='training architecture [Vgg19, Resnet50, Densenet121]')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())

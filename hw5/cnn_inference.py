import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.modules import Resnet50, Vgg19, Densenet121
from models.cnn_model import CNN
from utils.dataset import TrimmedVideo
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
    model = CNN(args).cuda() if with_cuda else CNN(args)
    model.load_state_dict(checkpoint['state_dict'])

    # dataset
    valid_dataset = TrimmedVideo(args, 'eval')
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                   batch_size=len(valid_dataset),
                                   collate_fn=valid_dataset.cnn_collate_fn,
                                   shuffle=False)
    # 1-3
    result, ground_truth = [], []
    with torch.no_grad():
        print("Predicting......")
        model.eval()

        for video, label in valid_data_loader:
            video = Variable(video).cuda() if with_cuda else Variable(video)
            output = model(video)
            value, index = torch.max(output, dim=1)
            result.append(index)
            ground_truth.append(label)

    result = result[0].cpu().data.numpy()
    ground_truth = ground_truth[0].numpy()

    accuracy = np.mean(result == ground_truth)
    print("Model accuracy:", accuracy)

    filename = os.path.join(output_file, 'p1_valid.txt')
    with open(filename, 'w') as f:
        result = [str(result[idx]) if idx == len(result)-1 else str(result[idx])+'\n' for idx in range(len(result))]
        f.writelines(result)

    # 1-2
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

    filename = os.path.join(output_file, 'fig1_2.jpg')
    plt.savefig(filename)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CNN inference")
    parser.add_argument('--input-feature', default='cnn_valid_feature.tar',
                        help='input feature file')
    parser.add_argument('--input-csv', default='HW5_data/TrimmedVideos/label/gt_valid.csv',
                        help='input csv file')
    parser.add_argument('--output-file', default='saved/cnn',
                        help='output data directory')
    parser.add_argument('--checkpoint', default='checkpoints/cnn_resnet50/epoch150_checkpoint.pth.tar',
                        help='load checkpoint')
    parser.add_argument('--pretrained', default='Resnet50', type=str,
                        help='training architecture [Vgg19, Resnet50]')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())

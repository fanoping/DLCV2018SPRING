from datasets.dataset import Cifar100
from model.relationnet import Relationnet
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import argparse
import json
import csv
import os


def main(args):
    config = json.load(open(args.config))
    checkpoint = torch.load(args.checkpoint)
    with_cuda = config['cuda']

    # test
    test_dataset = Cifar100(config=config,
                            mode='eval',
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))

    model = Relationnet(config).cuda() if with_cuda else Relationnet(config)
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        model.eval()
        label = test_dataset.novel_label
        results = []
        for _, (support_image, query_image) in enumerate(test_dataset):
            support_image = Variable(support_image).cuda() if with_cuda else Variable(support_image)
            query_image = Variable(query_image).cuda() if with_cuda else Variable(query_image)

            output = model(support_image, query_image)
            _, result = torch.max(output, dim=1)
            results.append(result)
        results = torch.cat(results, dim=0).data.cpu().numpy().tolist()
        results = [label[idx] for idx in results]

    filename = os.path.join('saved',
                            config['save']['dir'],
                            'inference.csv')
    with open(filename, 'w') as f:
        s = csv.writer(f, delimiter=',', lineterminator='\n')
        s.writerow(["image_id", "predicted_label"])
        for idx, predict_label in enumerate(results):
            s.writerow([idx, predict_label])
    print("Saving inference label csv as {}".format(filename))
    print("Inference done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument('--config', default='configs/relationnet_config.json')
    parser.add_argument('--checkpoint', required=True,
                        help='saved checkpoint file (*.tar)')
    main(parser.parse_args())

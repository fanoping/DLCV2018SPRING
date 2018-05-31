from models.modules.pretrained import *
import torch
import argparse
from tqdm import tqdm
import gc

def main(args):
    with_cuda = not args.no_cuda

    train_video = torch.load('train_video.tar')

    model = Resnet50().cuda() if with_cuda else Resnet50()
    model.eval()

    features = []
    with torch.no_grad():
        for name, video_frames in tqdm(train_video.items()):
            video_frames = video_frames.cuda() if with_cuda else video_frames
            extracted = model(video_frames)
            features.append(extracted.squeeze())



    print(len(features))

    torch.save(features, 'cnn_features.tar')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNN feature extractor')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')

    main(parser.parse_args())

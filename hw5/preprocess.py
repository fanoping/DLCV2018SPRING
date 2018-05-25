from utils.reader import getVideoList, readShortVideo
import torch
import argparse
import sys

def main(args):
    """
        preprocess of training/validation video frames
    """
    data_list = getVideoList(args.csv_file)
    all_frames = {}
    for i, (category, name) in enumerate(zip(data_list['Video_category'],
                                              data_list['Video_name'])):
        print("Loading: Category: {} Name: {}".format(category, name), end=' ')
        frames = readShortVideo(args.video_dir, category, name, 12, 1)
        print(len(frames), 'frames', end='\r')
        sys.stdout.write('\033[K')
        frames = torch.stack(frames, 0)
        all_frames[str(category+'/'+name)] = frames

    torch.save(all_frames, args.video_file)
    print('Frames of {} videos loaded.'.format(len(all_frames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess video data')
    parser.add_argument('--csv-file', default='HW5_data/TrimmedVideos/label/gt_train.csv',
                        help='action label csv file')
    parser.add_argument('--video-dir', default='HW5_data/TrimmedVideos/video/train',
                        help='video data for train/validation')
    parser.add_argument('--video-file', default='train_video.tar',
                        help='video file for dumping pickle')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())

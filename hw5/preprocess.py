from utils.reader import getVideoList, readShortVideo, readFullLengthVideo
import torch
import argparse
import sys
import os


def main(args):
    """
        preprocess of training/validation video frames (trimmed / full length)
    """
    if args.mode == 'trimmed':
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
        print('Frames of {} trimmed-videos loaded.'.format(len(all_frames)))

    elif args.mode == 'full-length':
        categories = sorted([file for file in os.listdir(args.full_length_dir) if file.startswith('OP')])

        full_length_frames = {}
        for category in categories:
            print("Loading: Category: {}".format(category), end=' ')
            frames = readFullLengthVideo(args.full_length_dir, category)
            print(len(frames), 'frames')
            frames = torch.stack(frames, 0)
            full_length_frames[category] = frames

        torch.save(full_length_frames, args.full_length_file)
        print('Frames of {} full-length videos loaded.'.format(len(full_length_frames)))

    else:
        return NotImplementedError('Choose either trimmed or full-length')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess video data')
    parser.add_argument('--mode', default='trimmed',
                        help='trimmed or full [trimmed / full]')
    parser.add_argument('--csv-file', default='HW5_data/TrimmedVideos/label/gt_train.csv',
                        help='action label csv file')
    parser.add_argument('--video-dir', default='HW5_data/TrimmedVideos/video/train',
                        help='video data for train/validation')
    parser.add_argument('--video-file', default='train_video.tar',
                        help='trimmed video file for dumping tar')
    parser.add_argument('--full-length-dir', default='HW5_data/FullLengthVideos/videos/train',
                        help='video data for train/validation')
    parser.add_argument('--full-length-file', default='train_full_length_video.tar',
                        help='full length video file for dumping tar')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())

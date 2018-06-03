import numpy as np
import skvideo.io
import skimage.io
import skimage.transform
import collections
import csv
import os
import torchvision.transforms as transforms


def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    """
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath, filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    for frameIdx, frame in enumerate(videogen):
        if frameIdx % downsample_factor == 0:
            frame = skimage.transform.rescale(frame,
                                              rescale_factor,
                                              mode='constant',
                                              preserve_range=True).astype(np.uint8)
            frame = transform(frame)
            frames.append(frame)
        else:
            continue
    return frames


def getVideoList(data_path):
    """
    @param data_path: ground-truth file path (csv files)

    @return: ordered dictionary of videos and labels
        {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
    """
    result = {}

    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column, value in row.items():
                result.setdefault(column, []).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od


def readFullLengthVideo(video_path, video_category):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    filepath = os.path.join(video_path, video_category)
    video_frames = sorted([os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('.jpg')])

    frames = []
    for frame_file in video_frames:
        frame = skimage.io.imread(frame_file)
        frame = transform(frame)
        frames.append(frame)

    return frames


def getLabelList(label_dir):
    files = sorted([os.path.join(label_dir, file) for file in os.listdir(label_dir) if file.endswith('.txt')])

    all_video_labels = []

    for file in files:
        with open(file, 'r') as f:
            lines = [int(line.strip()) for line in f.readlines()]
            all_video_labels.append(lines)

    return all_video_labels


if __name__ == '__main__':
    # readFullLengthVideo('../HW5_data/FullLengthVideos/videos/train', 'OP01-R01-PastaSalad')
    getLabelList('../HW5_data/FullLengthVideos/labels/train')

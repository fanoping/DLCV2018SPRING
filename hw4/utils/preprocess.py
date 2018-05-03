from scipy.misc import imread
import numpy as np
import os


def readfigs(filepath):
    """
        Image Reader
        pics shape: number of pics x image shape (64 x 64 x 3)
    """
    file_list = [file for file in os.listdir(filepath)]
    file_list.sort()

    pics = []
    for i, fig in enumerate(file_list):
        pic = imread(os.path.join(filepath, fig))
        pic = pic / 255.0
        pics.append(pic)
    pics = np.array(pics)
    pics = np.transpose(pics, (0, 3, 1, 2))

    return pics

if __name__ == '__main__':
    pics = readfigs('hw4_data/train')

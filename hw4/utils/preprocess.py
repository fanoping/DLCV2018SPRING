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
        pics.append(pic)
    pics = np.array(pics)

    print("{} data loaded.".format(len(pics)))

    return pics


if __name__ == '__main__':
    pics = readfigs('hw4_data/train')

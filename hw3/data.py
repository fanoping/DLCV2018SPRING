from PIL import Image
import numpy as np
import os
import scipy.misc


"""
    loading image
    sat:
        type: ndarray
        size: number of images (2313) x image size (512 x 512 x 3)   
    mask:
        type: ndarray
        size: number of images (2313) x image size (512 x 512)
    
    color segmentation
        urban       (0, 255, 255)
        agriculture (255, 255, 0)
        rangeland  (255, 0, 255)
        forest      (0, 255, 0)
        water       (0, 0, 255)
        barren      (255, 255, 255)
        unknown     (0, 0, 0)
"""


def read_sats(filepath):
    '''
        Read sats
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()

    sats = []
    for _, file in enumerate(file_list):
        sat = scipy.misc.imread(os.path.join(filepath, file))
        sats.append(sat)
    sats = np.array(sats, dtype='uint8')

    return sats

def read_masks(filepath):
    '''
    Read masks (from mean_iou)
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land
        masks[i, mask == 1] = 4  # (Blue: 001) Water
        masks[i, mask == 7] = 5  # (White: 111) Barren land
        masks[i, mask == 0] = 6  # (Black: 000) Unknown

    return masks


if __name__ == '__main__':
    train_y = read_masks('hw3-train-validation/train')
    print(train_y.shape)
    train_x =read_sats('hw3-train-validation/train')
    print (train_x.shape)
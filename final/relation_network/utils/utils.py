from scipy.misc import imread, imsave
import os


def listdir(directory, key=None):
    if key:
        return sorted(os.listdir(directory), key=key)
    else:
        return sorted(os.listdir(directory), key=lambda x: int(os.path.splitext(x)[0]))


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# image process (read, save, plot)
def read_image(image_dir, type='png'):
    image_files = [os.path.join(image_dir, image) for image in listdir(image_dir) if image.endswith(type)]
    image_files = [imread(image) for image in image_files]
    return image_files


def save_image(image, filename):
    imsave(filename, image)


if __name__ == '__main__':
    mkdir('output')
    image = read_image('../datasets/test')
    #save_image(image, 'output/test.png')

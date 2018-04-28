from data import read_sats
from keras.models import load_model
import argparse
import numpy as np
import os
from scipy.misc import imsave


def main(args):
    print("Inference")

    test_x = read_sats(args.input_dir) / 255.0
    if args.file == "FCN32s":
        if not os.path.exists("FCN32s/weights.20.hdf5"):
            print ("Have not trained.")
            return
        model = load_model("FCN32s/weights.20.hdf5")
    elif args.file == "FCN16s":
        if not os.path.exists("FCN16s/weights.20.hdf5"):
            print ("Have not trained.")
            return
        model = load_model("FCN16s/weights.20.hdf5")
    elif args.file == "FCN8s":
        if not os.path.exists("FCN8s/weights.20.hdf5"):
            print ("Have not trained.")
            return
        model = load_model("FCN8s/weights.20.hdf5")
    else:
        model = load_model(args.file)

    result = model.predict(test_x)
    result = np.argmax(result, axis=3)

    print ("Generating Figures")
    num = len(result)
    masks = np.empty((num, 512, 512, 3))
    for i, mask in enumerate(result):
        masks[i, mask == 0] = [0, 255, 255]  # (Cyan: 011) Urban land
        masks[i, mask == 1] = [255, 255, 0]  # (Yellow: 110) Agriculture land
        masks[i, mask == 2] = [255, 0, 255]  # (Purple: 101) Rangeland
        masks[i, mask == 3] = [0, 255, 0]  # (Green: 010) Forest land
        masks[i, mask == 4] = [0, 0, 255]  # (Blue: 001) Water
        masks[i, mask == 5] = [255, 255, 255]  # (White: 111) Barren land
        masks[i, mask == 6] = [0, 0, 0]  # (Black: 000) Unknown

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i, image in enumerate(masks):
        imsave(os.path.join(args.output_dir, "{0:04}_mask.png".format(i)), image)

    print("Inference Finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework 3')
    parser.add_argument('--input-dir', type=str,
                        help='testing data directory')
    parser.add_argument('--output-dir', type=str,
                        help='testing data output directory')
    parser.add_argument('--file', default='FCN32s', type=str,
                        help='file of the model [FCN32s, FCN16s, FCN8s] or other files [xxx.hdf5]')

    main(parser.parse_args())

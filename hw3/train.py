from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical
from data import read_masks, read_sats
from model import FCN32s, FCN16s, FCN8s
import argparse
import os

train_path = 'hw3-train-validation/train'
valid_path = 'hw3-train-validation/validation'


def train(args):
    print("Loading Training Files......")
    sat = read_sats(train_path) / 255.0
    mask = read_masks(train_path)

    print("Loading Validation Files......")
    sat_valid = read_sats(valid_path) / 255.0
    mask_valid = read_masks(valid_path)

    mask = to_categorical(mask, num_classes=7)
    mask_valid = to_categorical(mask_valid, num_classes=7)

    # optimizer
    adam = Adam(lr=0.0001)

    # model
    if args.arch == 'FCN32s':
        fcn = FCN32s(7)
    elif args.arch == 'FCN16s':
        fcn = FCN16s(7)
    elif args.arch == 'FCN8s':
        fcn = FCN8s(7)
    else:
        return NotImplementedError

    model = fcn.model
    print(fcn)

    # Checkpoint path
    if not os.path.exists(args.arch):
        os.makedirs(args.arch)

    callbacks = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    check_point = ModelCheckpoint(os.path.join(args.arch, 'weights.{epoch:02d}.hdf5'),
                                  verbose=1,
                                  save_best_only=False,
                                  save_weights_only=False,
                                  monitor='val_loss',
                                  mode='min')
    callbacks.append(early_stopping)
    callbacks.append(check_point)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(sat, mask, batch_size=args.batch_size, epochs=args.epochs,
              verbose=1, callbacks=callbacks, validation_data=(sat_valid, mask_valid))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework 3')
    parser.add_argument('--arch', default='FCN32s', type=str,
                        help='architecture of the model [FCN32s, FCN16s, FCN8s] (default: FCN32s)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='batch size (default: 8)')
    parser.add_argument('-e', '--epochs', default=20, type=int,
                        help='number of epochs (default: 20)')
    train(parser.parse_args())

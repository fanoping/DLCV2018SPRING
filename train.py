import keras
from keras.models import Input, Model
from keras.layers import Embedding, Conv2D, MaxPooling2D,AveragePooling2D,ZeroPadding2D, Flatten
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping ,ModelCheckpoint

import scipy.misc as misc
import numpy as np

train_path = 'datasets/hw3-train-validation/train/'
"""
    loading image
    sat:
        type: ndarray
        size: number of images (2313) x image size (512 x 512 x 3)   
    mask:
        type: ndarray
        size: number of images (2313) x image size (512 x 512 x 3)
"""
sat, mask = [], []
mask_files = ['{0:04}_mask.png'.format(i) for i in range(2313)]
sat_files = ['{0:04}_sat.jpg'.format(i) for i in range(2313)]
for sat_file in sat_files:
    sat.append(misc.imread(train_path+sat_file))
for mask_file in mask_files:
    mask.append(misc.imread(train_path+mask_file))
sat = np.array(sat)
mask = np.array(mask)

##### model

input_img = Input(shape=(512, 512, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

vgg_16 = Model(input_img, x)
vgg_16.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
vgg_16.summary()

o = UpSampling2D((2,2), name='block6_upsample')(x)
o = Conv2DTranspose(512, (3,3), activation='relu', padding='same', name='block6_conv_t1')(o)
o = Conv2DTranspose(512, (3,3), activation='relu', padding='same', name='block6_conv_t2')(o)

o = UpSampling2D((2,2), name='block7_upsample')(o)
o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='block7_conv_t1')(o)
o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='block7_conv_t2')(o)

o = UpSampling2D((2,2), name='block8_upsample')(o)
o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='block8_conv_t1')(o)
o = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='block8_conv_t2')(o)

o = UpSampling2D((2,2), name='block9_upsample')(o)
o = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='block9_conv_t1')(o)
o = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block9_conv_t2')(o)

o = UpSampling2D((2,2), name='block10_upsample')(o)
o = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block10_conv_t1')(o)
o = Conv2DTranspose(3, (3, 3), activation='relu', padding='same', name='block10_conv_t2')(o)


model = Model(input_img, o)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(sat, mask, batch_size=100, epochs=1, verbose=1)

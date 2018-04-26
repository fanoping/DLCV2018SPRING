from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import UpSampling2D, Conv2DTranspose, Dropout, Activation
from keras.layers import Add

class FCN32s:
    def __init__(self, classes):
        self.classes = classes
        self.build_model()

    def build_model(self):
        """
            vgg16,
                input:  3 x 512 x 512
                output: 512 x 16 x 16
            fcn32s
                input:  512 x 16 x 16
                output: 7 x 512 x 512
        """
        input_img = Input(shape=(512, 512, 3))
        x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
        x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x_1)
        x_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x_1)

        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x_2)
        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x_2)
        x_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x_2)

        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x_3)
        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x_3)
        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x_3)
        x_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x_3)

        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x_4)
        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x_4)
        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x_4)
        x_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x_4)

        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x_5)
        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x_5)
        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x_5)
        x_o = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x_5)

        self.vgg_16 = Model(input_img, x_o)
        self.vgg_16.load_weights('datasets/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

        o = Conv2D(4096, (3, 3), activation='relu', padding='same', name='block6_conv1')(x_o)
        o = Dropout(0.5)(o)
        o = Conv2D(4096, (1, 1), activation='relu', padding='same', name='block6_conv2')(o)
        o = Dropout(0.5)(o)

        o = Conv2D(self.classes, (1, 1), padding='valid', kernel_initializer='he_normal', name='score')(o)
        o = Conv2DTranspose(self.classes, (64, 64), strides=(32, 32), padding='same', activation='softmax',
                            use_bias=False, name='block6_upsample')(o)

        self.model = Model(input_img, o)

    def __str__(self):
        self.model.summary()
        return "Training on VGG16-FCN32s......"


class FCN16s:
    def __init__(self, classes):
        self.classes = classes
        self.build_model()

    def build_model(self):
        """
            vgg16,
                input:  3 x 512 x 512
                output: 512 x 16 x 16
            fcn16s
                input:  512 x 16 x 16
                output: 7 x 512 x 512
        """
        input_img = Input(shape=(512, 512, 3))
        x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
        x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x_1)
        x_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x_1)

        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x_1)
        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x_2)
        x_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x_2)

        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x_2)
        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x_3)
        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x_3)
        x_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x_3)

        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x_3)
        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x_4)
        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x_4)
        x_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x_4)

        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x_4)
        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x_5)
        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x_5)
        x_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x_5)

        self.vgg_16 = Model(input_img, x_5)
        self.vgg_16.load_weights('datasets/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

        o = Conv2D(4096, (3, 3), activation='relu', padding='same', name='block6_conv1')(x_5)
        o = Dropout(0.5)(o)
        o = Conv2D(4096, (1, 1), activation='relu', padding='same', name='block6_conv2')(o)
        o = Dropout(0.5)(o)

        o = Conv2D(self.classes, (1, 1), padding='valid', kernel_initializer='he_normal', name='score')(o)
        o = Conv2DTranspose(self.classes, (4, 4), strides=(2, 2), padding='valid', name='block6_upsample')(o)
        o = Cropping2D(cropping=((0, 2), (0, 2)))(o)

        pool4 = Conv2D(self.classes, (1, 1), padding='same', kernel_initializer='he_normal', name='score_pool4')(x_4)
        merge = Add()([o, pool4])
        o = Conv2DTranspose(self.classes, (32, 32), strides=(16, 16), padding='same', activation='softmax',
                            use_bias=False, name='block7_upsample')(merge)

        self.model = Model(input_img, o)

    def __str__(self):
        self.model.summary()
        return "Training on VGG16-FCN16s......"


class FCN8s:
    def __init__(self, classes):
        self.classes = classes
        self.build_model()

    def build_model(self):
        """
            vgg16,
                input:  3 x 512 x 512
                output: 512 x 16 x 16
            fcn8s
                input:  512 x 16 x 16
                output: 7 x 512 x 512
        """
        input_img = Input(shape=(512, 512, 3))
        x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
        x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x_1)
        x_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x_1)

        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x_1)
        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x_2)
        x_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x_2)

        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x_2)
        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x_3)
        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x_3)
        x_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x_3)

        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x_3)
        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x_4)
        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x_4)
        x_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x_4)

        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x_4)
        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x_5)
        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x_5)
        x_o = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x_5)

        self.vgg_16 = Model(input_img, x_o)
        self.vgg_16.load_weights('datasets/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

        o = Conv2D(4096, (3, 3), activation='relu', padding='same', name='block6_conv1')(x_o)
        o = Dropout(0.5)(o)
        o = Conv2D(4096, (1, 1), activation='relu', padding='same', name='block6_conv2')(o)
        o = Dropout(0.5)(o)

        o = Conv2D(self.classes, (1, 1), padding='valid', kernel_initializer='he_normal', name='score')(o)
        o = Conv2DTranspose(self.classes, (4, 4), strides=(2, 2), padding='valid', name='block6_upsample')(o)
        o = Cropping2D(cropping=((0, 2), (0, 2)))(o)

        pool4 = Conv2D(self.classes, (1, 1), padding='same', kernel_initializer='he_normal', name='score_pool4')(x_4)
        merge = Add()([o, pool4])
        o = Conv2DTranspose(self.classes, (4, 4), strides=(2, 2), padding='valid', name='block7_upsample')(merge)
        o = Cropping2D(cropping=((0, 2), (0, 2)))(o)

        pool3 = Conv2D(self.classes, (1, 1), padding='same', kernel_initializer='he_normal', name='score_pool3')(x_3)
        merge = Add()([o, pool3])
        o = Conv2DTranspose(self.classes, (16, 16), strides=(8, 8), padding='same', activation='softmax',
                            use_bias=False, name='block8_upsample')(merge)

        self.model = Model(input_img, o)

    def __str__(self):
        self.model.summary()
        return "Training on VGG16-FCN8s......"


if __name__ == '__main__':
    model = FCN32s(7)
    print(model)

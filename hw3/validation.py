import numpy as np
from PIL import Image
from keras.models import load_model


vgg_16 = load_model('vgg_16.h5')
model = load_model('weights.20.hdf5')
sat_valid = np.load('sat_validation.npy') / 255.0
result = model.predict(sat_valid)


color_label = np.array([[0, 255, 255],
               [255, 255, 0],
               [255, 0, 255],
               [0, 255, 0],
               [0, 0, 255],
               [255, 255, 255],
               [0, 0, 0]])

for i, fig in enumerate(result):
    fig_height = []
    for height in range(result.shape[1]):
        fig_width = []
        for width in range(result.shape[2]):
             fig_width.append(color_label[np.argmax(fig[height][width])])
        fig_height.append(fig_width)
    output = np.array(fig_height)
    im = Image.fromarray(output.astype('uint8'))
    im.save('results/{0:04}_mask.png'.format(i))
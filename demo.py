# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import img_rows, img_cols, num_classes, test_folder, original_images_key, gray_values
from model import build_model
from utils import get_best_model

if __name__ == '__main__':
    model = build_model()
    model.load_weights(get_best_model())

    print(model.summary())

    test_dir = os.path.join(test_folder, original_images_key)
    test_dir = os.path.join(test_dir, 'P0089_MacularCube512x128_4-25-2013_9-32-13_OD_sn2218_cube_z.img')
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith('.bmp')]
    samples = random.sample(test_images, 10)

    for i, filename in enumerate(samples):
        print('Start processing image: {}'.format(filename))

        image_bgr = cv.imread(filename)
        image_bgr = cv.resize(image_bgr, (img_cols, img_rows), cv.INTER_CUBIC)
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        x_test[0] = image_rgb / 127.5 - 1.

        out = model.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols, num_classes))
        out = np.argmax(out, axis=2)
        for j in range(num_classes):
            out[out == j] = gray_values[j]
        out = out.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), image_bgr)
        cv.imwrite('images/{}_out.png'.format(i), out)

    K.clear_session()

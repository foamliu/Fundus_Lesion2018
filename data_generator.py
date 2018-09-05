import json

import cv2 as cv
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical

from config import img_rows, img_cols, batch_size, num_classes


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        if self.usage == 'train':
            gt_file = 'train_gt_file.json'
        else:
            gt_file = 'valid_gt_file.json'

        with open(gt_file, 'r') as file:
            self.samples = json.load(file)

        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        X = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        Y = np.empty((length, img_rows, img_cols, num_classes), dtype=np.float32)

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            name = self.names[id]
            image = get_image(name)
            category = get_category(id)
            image, category = random_crop(image, category)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            X[i_batch] = image
            Y[i_batch] = to_categorical(category, num_classes)

        X = preprocess_input(X)

        return X, Y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')

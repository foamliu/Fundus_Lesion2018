import cv2 as cv
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical

from config import img_rows, img_cols, batch_size
from config import num_classes


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        with open('{}_ids.txt'.format(usage), 'r') as f:
            ids = f.read().splitlines()
            self.ids = list(map(int, ids))

        with open('names.txt', 'r') as f:
            self.names = f.read().splitlines()

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.ids) - i))
        X = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        Y = np.empty((length, img_rows, img_cols, num_classes), dtype=np.float32)

        for i_batch in range(length):
            id = self.ids[i + i_batch]
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
        np.random.shuffle(self.ids)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')

import json

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import img_rows, img_cols, batch_size, num_classes, gray_values


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
        Y = np.empty((length, img_rows, img_cols), dtype=np.int32)

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            original_image_path = sample['original_image']
            label_image_path = sample['label_image']
            original_image = cv.imread(original_image_path)
            original_image = cv.resize(original_image, (img_rows, img_cols), cv.INTER_NEAREST)
            label_image = cv.imread(label_image_path, 0)
            label_image = cv.resize(label_image, (img_rows, img_cols), cv.INTER_NEAREST)
            for i in range(num_classes):
                label_image[label_image == gray_values[i]] = i

            X[i_batch] = original_image / 127.5 - 1.
            Y[i_batch] = label_image

        return X, Y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')

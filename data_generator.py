import json

import cv2 as cv
import numpy as np
from keras.utils import Sequence, to_categorical

from augmentor import seq_det, seq_img
from config import img_rows, img_cols, batch_size, num_classes, gray_values
from utils import preprocess_input


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
        Y = np.zeros((length, img_rows, img_cols, num_classes), dtype=np.float32)

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            original_image_path = sample['original_image']
            label_image_path = sample['label_image']
            original_image = cv.imread(original_image_path)
            original_image = cv.resize(original_image, (img_cols, img_rows), cv.INTER_NEAREST)
            label_image = cv.imread(label_image_path, 0)
            label_image = cv.resize(label_image, (img_cols, img_rows), cv.INTER_NEAREST)

            X[i_batch] = original_image
            for j in range(num_classes):
                Y[i_batch][label_image == gray_values[j]] = to_categorical(j, num_classes)

        # if self.usage == 'train':
        #     X = seq_img.augment_images(X)
        #     X = seq_det.augment_images(X)
        #     Y = seq_det.augment_images(Y)

        X = preprocess_input(X)

        return X, Y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)


def revert_labeling(y):
    y = np.argmax(y, axis=-1)
    for j in range(num_classes):
        y[y == j] = gray_values[j]
    return y


if __name__ == '__main__':
    data_gen = DataGenSequence('train')
    item = data_gen.__getitem__(0)
    x, y = item
    print(x.shape)
    print(y.shape)

    for i in range(10):
        image = revert_pre_process(x[i])
        h, w = image.shape[:2]
        image = image[:, :, ::-1].astype(np.uint8)
        print(image.shape)
        cv.imwrite('images/sample_{}.jpg'.format(i), image)
        label = revert_labeling(y[i])
        label = label.astype(np.uint8)
        cv.imwrite('images/label_{}.jpg'.format(i), label)

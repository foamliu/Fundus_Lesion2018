# -*- coding: utf-8 -*-
import os
import zipfile

import cv2 as cv
from tqdm import tqdm

from config import train_folder, label_images_key


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def check_gt_file():
    label_images_root = os.path.join(train_folder, label_images_key)
    label_images_folders = [f for f in os.listdir(label_images_root) if
                            os.path.isdir(os.path.join(label_images_root, f))]

    pixel_dict = dict()
    for v in range(256):
        pixel_dict[v] = 0

    for label_images_folder in tqdm(label_images_folders):
        parent = os.path.join(label_images_root, label_images_folder)
        files = [f for f in os.listdir(parent) if f.lower().endswith('.bmp')]

        for file in files:
            label_path = os.path.join(parent, file)
            image = cv.imread(label_path, 0)
            height, width = image.shape[:2]
            for i in range(height):
                for j in range(width):
                    pixel_dict[image[i, j]] += 1

    for v in range(256):
        count = pixel_dict[v]
        if count > 0:
            print('{} -> {}'.format(v, count))


if __name__ == '__main__':
    check_gt_file()

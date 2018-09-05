# -*- coding: utf-8 -*-
import os
import zipfile
import json

from config import train_folder, valid_folder, original_images_key, label_images_key

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def gen_gt_file(folder, usage):
    original_images_root = os.path.join(folder, original_images_key)
    folders = [f for f in os.listdir(original_images_root) if os.path.isdir(os.path.join(original_images_folder, f))]
    label_images_root = os.path.join(folder, label_images_key)

    gt_list = []
    for folder in folders:
        parent = os.path.join(original_images_root, folder)
        files = [f for f in os.listdir(parent) if f.lower().endswith('.bmp')]

        for file in files:
            original_path = os.path.join(parent, file)
            label_path = os.path.join(os.path.join(label_images_root, folder), file)
            if not os.path.isfile(label_path):
                print('cannot find: {}'.format(label_path))
            item = {'original_image': original_path, 'label_path': label_path}
            gt_list.append(item)
            print(item)

    with open('{}_gt_file.json'.format(usage), 'w') as file:
        json.dump(gt_list, file, indent=4)


if __name__ == '__main__':
    ensure_folder('data')

    extract('ai_challenger_fl2018_trainingset')
    extract('ai_challenger_fl2018_validationset')
    extract('ai_challenger_fl2018_testset')

    gen_gt_file(train_folder, 'train')
    gen_gt_file(valid_folder, 'valid')

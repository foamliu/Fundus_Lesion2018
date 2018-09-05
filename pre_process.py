# -*- coding: utf-8 -*-
import os
import zipfile


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    ensure_folder('data')

    extract('ai_challenger_fl2018_trainingset')
    extract('ai_challenger_fl2018_validationset')
    extract('ai_challenger_fl2018_testset')

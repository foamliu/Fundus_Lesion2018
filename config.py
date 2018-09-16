img_rows, img_cols = 512, 256
channel = 3
batch_size = 16
epochs = 1000
patience = 50
num_samples = 10880
num_train_samples = 8960
# num_samples - num_train_samples
num_valid_samples = 1920
num_classes = 4
kernel = 3
weight_decay = 1e-2

train_folder = 'data/Edema_trainingset/'
valid_folder = 'data/Edema_validationset/'
test_folder = 'data/Edema_testset/'
original_images_key = 'original_images'
label_images_key = 'label_images'

labels = ['Background', 'PED', 'SRF', 'REA']
gray_values = [0, 128, 191, 255]

prior_prob = [0.] * 4
num_total_pixels = 4365881383 + 1295967 + 39616122 + 290827008
prior_prob[0] = 4365881383 / num_total_pixels
prior_prob[1] = 1295967 / num_total_pixels
prior_prob[2] = 39616122 / num_total_pixels
prior_prob[3] = 290827008 / num_total_pixels
import numpy as np
prior_prob = [9.29380711e-01, 2.76152904e-04, 8.43349828e-03, 6.19096379e-02]
prior_prob_smoothed = [0.90058363, 0.01290955, 0.01695968, 0.06954713]
prior_factor = np.array([0.78088884, 3.41744114, 3.36559402, 2.81172266])

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
from utils import smooth_color_prior, compute_prior_factor

prior_prob_smoothed = smooth_color_prior(prior_prob, sigma=5)
prior_factor = compute_prior_factor(prior_prob_smoothed, gamma=0.5, alpha=1)

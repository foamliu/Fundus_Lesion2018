import multiprocessing
import os

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve
from tensorflow.python.client import device_lib

from config import prior_factor, num_classes


def categorical_crossentropy_with_class_rebal(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, num_classes))
    y_pred = K.reshape(y_pred, (-1, num_classes))

    idx_max = K.argmax(y_true, axis=1)
    weights = K.gather(prior_factor, idx_max)
    weights = K.reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    cross_ent = K.mean(cross_ent, axis=-1)

    return cross_ent


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def preprocess_input(image):
    return image / 127.5 - 1.


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.

    y_true is a 3-rank tensor with the desired output.
    The shape is [batch_size, img_rows, img_cols].

    y_pred is the decoder's output which is a 4-rank tensor
    with shape [batch_size, img_rows, img_cols, num_labels]
    so that for each image in the batch there is a one-hot
    encoded array of length num_labels.
    """

    # Calculate the loss. This outputs a
    # 3-rank tensor of shape [batch_size, img_rows, img_cols]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 3-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


def get_best_model():
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_loss>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    files = [f for f in os.listdir('models/') if p.match(f)]
    filename = None
    if len(files) > 0:
        accs = [float(p.match(f).groups()[1]) for f in files]
        best_index = int(np.argmax(accs))
        filename = os.path.join('models', files[best_index])
    print('loading best model: {}'.format(filename))
    return filename


def smooth_color_prior(prior_prob, sigma=5):
    # add an epsilon to prior prob to avoid 0 vakues and possible NaN
    prior_prob += 1E-3 * np.min(prior_prob)
    # renormalize
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Smooth with gaussian
    f = interp1d(np.arange(prior_prob.shape[0]), prior_prob)
    xx = np.linspace(0, prior_prob.shape[0] - 1, 1000)
    yy = f(xx)
    window = gaussian(2000, sigma)  # 2000 pts in the window, sigma=5
    smoothed = convolve(yy, window / window.sum(), mode='same')
    fout = interp1d(xx, smoothed)
    prior_prob_smoothed = np.array([fout(i) for i in range(prior_prob.shape[0])])
    prior_prob_smoothed = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    return prior_prob_smoothed


def compute_prior_factor(prior_prob_smoothed, gamma=0.5, alpha=1):
    u = np.ones_like(prior_prob_smoothed)
    u = u / np.sum(1.0 * u)

    prior_factor = (1 - gamma) * prior_prob_smoothed + gamma * u
    prior_factor = np.power(prior_factor, -alpha)

    # renormalize
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))

    return prior_factor


def get_highest_acc():
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_acc>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    acces = [float(p.match(f).groups()[1]) for f in os.listdir('models/') if p.match(f)]
    if len(acces) == 0:
        import sys
        return sys.float_info.min
    else:
        return np.max(acces)
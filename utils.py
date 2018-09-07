import multiprocessing
import os

import cv2 as cv
import tensorflow as tf
from tensorflow.python.client import device_lib


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


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
        losses = [float(p.match(f).groups()[1]) for f in files]
        best_index = int(np.argmin(losses))
        filename = os.path.join('models', files[best_index])
    return filename
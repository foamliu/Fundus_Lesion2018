import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from config import patience, epochs, num_train_samples, num_valid_samples, batch_size
from data_generator import train_gen, valid_gen
from model import build_model
from utils import get_available_gpus, ensure_folder, get_best_model, get_highest_acc, \
    categorical_crossentropy_with_class_rebal

if __name__ == '__main__':
    checkpoint_models_path = 'models/'
    ensure_folder(checkpoint_models_path)
    pretrained_path = get_best_model()

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5, patience=int(patience / 4), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            highest_acc = get_highest_acc()
            if float(logs['val_acc']) > highest_acc:
                self.model_to_save.save(fmt % (epoch, logs['val_acc']))


    # Load our model, added support for Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = build_model()
            if pretrained_path is not None:
                model.load_weights(pretrained_path)

        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        new_model = build_model()
        if pretrained_path is not None:
            new_model.load_weights(pretrained_path)

    # sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5.)
    # new_model.compile(optimizer=sgd, loss=categorical_crossentropy_with_class_rebal, metrics=['accuracy'])
    new_model.compile(optimizer='adam', loss=categorical_crossentropy_with_class_rebal, metrics=['accuracy'])

    print(new_model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, reduce_lr, early_stop]

    # Start Fine-tuning
    new_model.fit_generator(train_gen(),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=False
                            )

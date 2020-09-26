# coding=utf-8
# Created by model.py on 2020-04-09 09:52
# Copyright © 2020 Alan. All rights reserved.

import h5py
import tensorflow as tf
from data import create_dataset, save_info
from config import Config
from model import get_model
from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,  save_freq='epoch')

f = h5py.File('./trimmed_dataset.hdf', 'r')
x_train, x_test, y_train, y_test = create_dataset(f, split=0.2)

print(y_test.shape, y_test)
# _ = get_streaming()
training_model = get_model()


def scheduler(epoch):
    return Config['learning_rate'] * tf.math.exp(- 0.1 * epoch)


lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
callback_list = [checkpoint, lr_callback]
results = training_model.fit(x_train, y_train, validation_data=(x_test, y_test),
                             batch_size=Config['batch_size'],
                             epochs=Config['epoch_num'],
                             callbacks=callback_list)

training_model.save_weights('weights.h5', overwrite=True)
save_info(training_model, results)
f.close()

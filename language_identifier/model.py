# coding=utf-8
# Created by model.py on 2020-04-09 09:52
# Copyright © 2020 Alan. All rights reserved.

from tensorflow.keras.layers import Input, Dense, GRU, TimeDistributed, Dropout, Flatten
from tensorflow.keras import Model
import tensorflow as tf
from config import Config


def get_model():
    spec = Input(batch_shape=(None, Config['train_seq_length'], Config['feature_dim']))
    x = spec
    x = GRU(128, return_sequences=True, stateful=False)(x)
    x = GRU(64, return_sequences=True, stateful=False)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32)(x)
    pred = Dense(3, activation='softmax')(x)
    model = Model(inputs=spec, outputs=pred)
    adam = tf.keras.optimizers.Adam(learning_rate=Config['learning_rate'])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    model.summary()
    return model


def get_streaming():
    streaming = Input(batch_shape=(1, None, Config['feature_dim']))
    x = streaming
    x = GRU(128, return_sequences=True, stateful=False)(x)
    x = GRU(64, return_sequences=True, stateful=False)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32)(x)
    pred = Dense(3, activation='softmax')(x)
    streaming_model = Model(inputs=streaming, outputs=pred)
    adam = tf.keras.optimizers.Adam(learning_rate=Config['learning_rate'])
    streaming_model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    streaming_model.summary()
    return streaming_model

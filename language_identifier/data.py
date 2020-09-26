# coding=utf-8
# Created by model.py on 2020-04-09 09:52
# Copyright © 2020 Alan. All rights reserved.

import os
import numpy as np
import pickle
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
from config import Config
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

languages = ['english', 'hindi', 'mandarin']
label = {'english': 0, 'hindi': 1, 'mandarin': 2}


def create_dataset(f, split=0.2):
    def preprocess(language, f):
        curr = f[language]
        num_sequence = curr.shape[0]
        dim_0 = num_sequence // Config['train_seq_length']
        curr = curr[:dim_0 * Config['train_seq_length'], :]
        curr = curr.reshape(dim_0, Config['train_seq_length'], Config['feature_dim'])
        l = np.array(curr.shape[0] * [curr.shape[1] * [label[language]]])
        l = l.reshape(curr.shape[0], curr.shape[1], 1)
        return curr, l

    x_0, y_0 = preprocess('english', f)
    x_2, y_2 = preprocess('mandarin', f)
    x_1, y_1 = preprocess('hindi', f)
    x = np.concatenate((x_0, x_1, x_2), axis=0)
    y = np.concatenate((y_0, y_1, y_2), axis=0)
    # arr = np.random.permutation(x.shape[0])
    # x = x[arr,:,:]
    # y = y[arr,:,:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, to_categorical(y_train,num_classes=3), to_categorical(y_test,num_classes=3)

def save_info(model, results):
    png_name = 'plot.png'
    pickle_name = 'history.pickle'

    if os.path.exists(Config['logs']):
        os.rmdir(Config['logs'])
    os.mkdir(Config['logs'])
    plot_model(model, to_file=os.path.join(Config['logs'], png_name))

    if os.path.exists(os.path.join(Config['logs'], pickle_name)):
        os.remove(os.path.join(Config['logs'], pickle_name))
    with open(os.path.join(Config['logs'], pickle_name), 'wb') as f:
        pickle.dump(results.history, f)

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

import os
import numpy as np
from PIL import Image

import tensorflow as tf
from utils import Config
import pickle


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()

    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self, train=True):
        # map id to category
        id_to_category = {}
        file = os.path.join(self.root_dir, Config['pairwise_train'])
        with open(file, 'r') as file_read:
            line = file_read.readline()
            while line:
                ori_line = line.split()
                id_to_category[ori_line[1]+'.jpg' + ' ' + ori_line[2]+'.jpg'] = int(ori_line[0])
                line = file_read.readline()

        file = os.path.join(self.root_dir, Config['pairwise_valid'])
        with open(file, 'r') as file_read:
            line = file_read.readline()
            while line:
                ori_line = line.split()
                id_to_category[ori_line[1]+'.jpg' + ' ' + ori_line[2]+'.jpg'] = int(ori_line[0])
                line = file_read.readline()

        X, y = [], []
        for key, value in id_to_category.items():
            X.append(key)
            y.append(value)
        y = LabelEncoder().fit_transform(y)

        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # train_limit = min(Config['max_size'], int(Config['proportion'] * len(X_train)))

        train_limit = int(Config['proportion'] * len(X_train))
        test_limit = int(0.25 * train_limit)

        return X_train[:train_limit], X_test[:test_limit], y_train[:train_limit], y_test[:test_limit], max(y) + 1


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, dataset_sizedataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = os.path.join(Config['root_path'], 'images')
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        X0, X1, y = self.__data_generation(indexes)
        X0, X1, y = np.stack(X0), np.stack(X1), np.stack(y)
        # print(X0.shape, X1.shape, y.shape)
        return (np.moveaxis(X0, 1, 3), np.moveaxis(X1, 1, 3)), y

    def __data_generation(self, indexes):
        X0 = []; X1 = []; y = []
        for idx in indexes:
            left_file, right_file = self.X[idx].split()
            file_path_0 = os.path.join(self.image_dir, left_file)
            file_path_1 = os.path.join(self.image_dir, right_file)
            X0.append(self.transform(Image.open(file_path_0)))
            X1.append(self.transform(Image.open(file_path_1)))
            y.append(self.y[idx])
        return X0, X1, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def save_info(model, results):
    model_name = 'model.hdf5'
    png_name = 'plot.png'
    pickle_name = 'history.pickle'

    if os.path.exists(Config['logs']):
        os.rmdir(Config['logs'])
    os.mkdir(Config['logs'])

    plot_model(model, to_file=os.path.join(Config['logs'], png_name))

    if os.path.exists(os.path.join(Config['logs'], model_name)):
        os.remove(os.path.join(Config['logs'], model_name))
    model.save(os.path.join(Config['logs'], model_name))

    if os.path.exists(os.path.join(Config['logs'], pickle_name)):
        os.remove(os.path.join(Config['logs'], pickle_name))
    with open(os.path.join(Config['logs'], pickle_name), 'wb') as f:
        pickle.dump(results.history, f)

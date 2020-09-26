from tensorflow.keras.models import Model, load_model

from data import polyvore_dataset, DataGenerator
from utils import Config
import tensorflow as tf
from model import siamese_mobilenet_v2
from data import save_info

# set train or restore training from existing model
if __name__ == '__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(X_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator = DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

    if Config['restore']:
        # restore training
        model = load_model(Config['restore_file'])
    else:
        pre_model, model = siamese_mobilenet_v2()

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=Config['learning_rate']),
        loss="binary_crossentropy",
        metrics=['acc', 'mse'])

    model.summary()

    results = model.fit_generator(
        generator=train_generator,
        validation_data=test_generator,
        epochs=Config['num_epochs'])

    save_info(model, results)

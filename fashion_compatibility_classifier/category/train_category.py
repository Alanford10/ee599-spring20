from tensorflow.keras.models import load_model
from tensorflow.keras import applications

from data import polyvore_dataset, DataGenerator
from utils import Config

from model import my_model
from data import save_info

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
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator = DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

    if Config['finetune']:
        # finetune models from imagenet model
        if Config['finetune'] == 'Xception':
            model = applications.xception.Xception(include_top=True, weights='imagenet', n_classes=n_classes)
        if Config['finetune'] == 'VGG16':
            model = applications.vgg16.VGG16(include_top=True, weights='imagenet', n_classes=n_classes)
        if Config['finetune'] == 'MobileNet':
            model = applications.mobilenet.MobileNet(include_top=True, weights='imagenet', n_classes=n_classes)
        if Config['finetune'] == 'ResNet50':
            model = applications.resnet.ResNet50(include_top=True, weights='imagenet', n_classes=n_classes)
        if Config['finetune'] == 'InceptionV3':
            model = applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', n_classes=n_classes)
    else:
        if Config['restore']:
            # restore training
            model = load_model(Config['restore_file'])
        else:
            # training from scratch
            model = my_model(classes=n_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    results = model.fit_generator(
                        generator=train_generator,
                        validation_data=test_generator,
                        epochs=Config['num_epochs']
                        )
    save_info(model, results)

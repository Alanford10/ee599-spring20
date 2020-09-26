Config = {}

# you should replace it with your own root_path
Config['root_path'] = '../polyvore_outfits/'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['logs'] = './logs'

# support model: 'Xception', 'VGG16', 'MobileNet', 'ResNet50', 'InceptionV3'.
# Config['finetune'] = 'VGG16'
Config['finetune'] = ''
Config['use_cuda'] = True
Config['restore'] = False
Config['restore_file'] = 'categorical-model_2.hdf5'
Config['debug'] = False
Config['num_epochs'] = 10
Config['batch_size'] = 64

Config['learning_rate'] = 0.001

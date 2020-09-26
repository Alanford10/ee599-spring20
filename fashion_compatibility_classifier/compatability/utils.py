Config = {}


Config['root_path'] = '../polyvore_outfits/'
Config['pairwise_valid'] = 'pairwise_compatibility_valid.txt'
Config['pairwise_train'] = 'pairwise_compatibility_train.txt'
Config['logs'] = './logs'

# whole set is too large for training
# only min(Config['max_size'], Config['proportion']*whole_set) of samples are chosen
Config['max_size'] = 200000
Config['proportion'] = 0.2

Config['use_cuda'] = True
Config['debug'] = False
Config['restore'] = False
Config['num_epochs'] = 13
Config['batch_size'] = 64

Config['learning_rate'] = 0.01

import numpy as np
import glob, os
import argparse
import librosa
import h5py


def wav_to_npy(filepath, dataset_name):
    fileList = os.listdir(filepath)
    dataset = None
    for filename in fileList:
        if filename[-4:] != '.wav':
            continue
        print('{} is processing'.format(filename))
        data, sr = librosa.load(os.path.join(filepath, filename), sr=16000)
        # sampling windows and hop length can be changed
        data, _ = librosa.effects.trim(data, top_db=10)
        data = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=64,
                                    n_fft=int(sr * 0.025), # it was 0.0025
                                    hop_length=int(sr*0.010)).T # kind of ml engieering mistake:)

        # with h5py.File(hdf_name) as f:
        # data_set = f.create_dataset(dataset_name, data=data, dtype="float32")
        if dataset is None:
            dataset = data
        else:
            dataset = np.append(data, dataset, axis=0)
        print(dataset.shape)
        print('{} completed'.format(filename))
    np.save(dataset_name, dataset)

language_list = ['english','mandarin','hindi']
"""
for language in language_list:
    input_dir = './dataset/train_' + language
    dataset_name = language + '.npy'
    wav_to_npy(input_dir, language + '.npy')
"""
f = h5py.File('trimmed_dataset.hdf', 'w')
f.create_dataset('english', data=np.load('english.npy'))
f.create_dataset('hindi', data=np.load('hindi.npy'))
f.create_dataset('mandarin', data=np.load('mandarin.npy'))

f.close()

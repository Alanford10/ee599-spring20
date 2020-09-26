import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras import backend

from torchvision import transforms

from PIL import Image
import os, json
from itertools import combinations 

# PREDICT_FROM_SET: predicts from a fashion set
# e.g. 224930161_1 224930161_2 224930161_3 224930161_4
# PREDICT_FROM_SET: predicts from a fashion pair
# e.g. 154249722 188425631

PREDICT_FROM_SET = False
threshold = 0.05
model_path = 'compatibility-model.hdf5'
root_dir = '../polyvore_outfits'
test_file = os.path.join(root_dir, 'test_pairwise_compat_hw.txt')
out_file = os.path.join('test_pairwise_compat_output.txt')
img_dir = os.path.join(root_dir, 'images')
meta_file = open(os.path.join(root_dir, 'test.json'), 'r')
meta_json = json.load(meta_file)
set_to_items = {}
compose = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def read_img(img_name, dir=img_dir, comp=compose):
    img = os.path.join(dir, img_name)
    img = Image.open(img)
    return comp(img)


for element in meta_json:
    set_to_items[element['set_id']] = element['items'] 

if os.path.exists(os.path.join(out_file)):
    os.remove(os.path.join(out_file))

file_write = open(os.path.join(out_file), 'w')

model = load_model(model_path, custom_objects={"backend": backend})

with open(os.path.join(root_dir, test_file), 'r') as file_read:
    line = file_read.readline()
    while line:
        outfit = line.split()
        if PREDICT_FROM_SET:
            comb = list(combinations(list(range(0, len(outfit))), 2))
            left, right = [], []
            for pair in comb:
                set1, idx1 = outfit[pair[0]].split('_')
                set2, idx2 = outfit[pair[1]].split('_')
                img1 = set_to_items[set1][int(idx1)-1]['item_id']
                img2 = set_to_items[set2][int(idx2)-1]['item_id']

                left.append(read_img(img_name=img1+'.jpg'))
                right.append(read_img(img_name=img2+'.jpg'))
            left, right = np.stack(left), np.stack(right)
            pred = model.predict((np.moveaxis(left, 1, 3), np.moveaxis(right, 1, 3)))

            pred_f = round(float(np.mean(pred)), 4)
            pred = str(pred_f)

            if pred_f > 0.5 + threshold:
                pred = 'Positive ' + pred
            elif pred_f < 0.5 - threshold:
                pred = 'Negative ' + pred
            else:
                pred = 'Undecided ' + pred
            file_write.write(pred + ' ' + line)

        else:
            img1, img2 = np.array(read_img(outfit[0]+'.jpg')), np.array(read_img(outfit[1]+'.jpg'))
            img1, img2 = img1[np.newaxis,:], img2[np.newaxis,:]
            pred = model.predict((np.moveaxis(img1, 1, 3), np.moveaxis(img2, 1, 3)))

            s = str(round(float(pred), 4))
            file_write.write(line[:-1] + ' ' + s + '\n')

        line = file_read.readline()

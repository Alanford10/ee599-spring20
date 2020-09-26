import numpy as np
import os
import json

from tensorflow.keras.models import load_model
from torchvision import transforms
from PIL import Image

root_dir = '../polyvore_outfits'
meta_file = 'polyvore_item_metadata.json'
model_path = 'categorical-model_2.hdf5'
test_file = 'test_category_hw.txt'
out_file = 'test_category_output.txt'
image_dir = os.path.join(root_dir, 'images')
meta_file = open(os.path.join(root_dir, meta_file), 'r')
meta_json = json.load(meta_file)
files = os.listdir(image_dir)

if os.path.exists(os.path.join(out_file)):
    os.rmdir(os.path.join(out_file))
file_write = open(os.path.join(out_file), 'w')

# input(1, 224, 224, 3) output(153)

model = load_model(model_path)
with open(os.path.join(root_dir, test_file), 'r') as file_read:
    line = file_read.readline()
    while line:
        name = line[:-1] + '.jpg'
        if name in files:
            # read image and transorm
            img_path = os.path.join(image_dir, name)
            img = Image.open(img_path)
            compose = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img = compose(img)
            img = np.array(img)
            img = img[np.newaxis, :]
            img = np.moveaxis(img, 1, 3)

            pred = model.predict(img)
            pred = str(np.argmax(pred))
            file_write.write(line[:-1] + ' ' + pred + '\n')
        line = file_read.readline()

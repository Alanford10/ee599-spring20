# Outfit Category and Compatibility Prediction

The goal of this assignment is to apply deep learning to computer vision. Particularly, you’ll work with the classiﬁcation problem on a fashion compatibility dataset called Polyvore. Your goals will be to set up your category classiﬁer and fashion compatibility classiﬁer. Turn in your code and report as described in Section 5.
The starter code can be found [here](https://github.com/davidsonic/EE599-CV-Project)

## Functions

* [category](./category): Categorical classfication. Train & predict
* [compatability](./compatability): Binary classfication for 2 image inputs, Train & predict compatibility 2 outfits and compatibility of mulitiple synchronized outfits set
* [read_history.py](./read_history.py: Read saved training scheme from log and plot. The scitpts may have several changes due to the parameter to read from the pickle file.
* [parser.py](./parser.py): To turn `compatibility_train.txt` and `compatibility_valid.txt` ( format: 1 210750761_1 210750761_2 210750761_3)into easy-to-read txt file(format: 1 154249722 188425631)
* [hw4](./hw4.pdf): Homework reqirement


## Usage

1. Download the original Polyvore dataset..
```
gdown https://drive.google.com/uc?id=1ZCDRRh4wrYDq0O3FOptSlBpQFoe0zHTw
```
Unzip the file and combine with  `polyvore_outfits`

2. Train/Transfer learning/finetune
Adjust `utils.py`
   ```sh
   python train.py
   ```
The Traing scheme and model will be preserved under ./log file if you train the model.

3. Evaluate
   ```
   python predict.py
   ```
**There might be several small changes for detailed file path**

## Sample Output

| function  | Input | Output|
| :----- | :------: | :-------: |
|category | 196184259 | 196184259 46    |
|pairwise compatability| 126827417 125676552 | 126827417 125676552 0.6726 |
|set compatability |209896910_1 209896910_2 209896910_3 | Positive 0.6824 209896910_1 209896910_2 209896910_3 |

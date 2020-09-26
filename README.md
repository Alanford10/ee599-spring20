# ee599-spring20
USC EE 599 course projects

## General Dependencies
python >= 3.7
1. pytorch/tensorflow>=2.0
2. tqdm
3. sklearn
4. torchvision
5. sklearn
6. librosa

## CV Project Assignment

This project has two subtasks: category classification and pairwise compatibility classification.
The first task requires construcing <img, class_label> pair for classification.
The second task requires constructing <(x1,x2), binary_label> pair for classification.

Dataset:
- Polyvore Outfits (download link): https://drive.google.com/open?id=1ZCDRRh4wrYDq0O3FOptSlBpQFoe0zHTw

[My CV implementation](./fashion_compatibility_classifier) and [document](./fashion_compatibility_classifier/README.md)

## LID Assignment
Spoken Language Identiﬁcation (LID) is broadly deﬁned as recognizing the language of a given speech utterance [1]. It has numerous applications in automated language and speech recognition, multilingual machine translations, speech-to-speech translations, and emergency call routing. In this homework, we will try to classify three languages (English, Hindi and Mandarin) from the spoken utterances that have been crowd-sourced from the class.

[My LID implementation](./language_identifier) and [document](./language_identifier/README.md)

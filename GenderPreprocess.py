import os
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

women_faces = os.listdir('Resources/GenderCNN/women') ## store directory containing pictures of women
men_faces = os.listdir('Resources/GenderCNN/men') ## store directory containing pictures of men

women = [] ## list to hold women pictures
men = [] ## list to hold men pictures

## loop through each image in directory
for woman in women_faces:
    if (woman[-4:] == '.jpg') & ('(' not in woman):
        women.append(woman) ## add jpg images and ones without parentheses in name to list

## loop through each image in directory
for man in men_faces:
    if (man[-4:] == '.jpg') & ('(' not in man):
        men.append(man) ## add jpg images and ones without parentheses in name to list

data = [] ## list to hold image data
labels = [] ## list to hold woman/man labels
w, h = 96, 96 ## standard size of image

women_path = ('Resources/GenderCNN/women/')
men_path = ('Resources/GenderCNN/men/')

## loop through each picture in women list to resize, convert to array, and add to data and labels lists
for pic in women:
    path = women_path+pic
    img = cv2.imread(path)
    img = cv2.resize(img, (w, h))
    img = img_to_array(img)
    data.append(img)
    labels.append(0)

## loop through each picture in men list to resize, convert to array, and add to data and labels lists
for pic in men:
    path = men_path+pic
    img = cv2.imread(path)
    img = cv2.resize(img, (w, h))
    img = img_to_array(img)
    data.append(img)
    labels.append(1)

data = np.array(data, dtype='float') / 255.0 ## convert image data numpy array of float values and divide by 255 to normalize values
labels = np.array(labels) ## convert labels array to numpy array

idx = np.random.permutation(len(data)) ## create randomized list of indices
x, y = data[idx], labels[idx] ## randomize the data and labels for training and rename as x and y

(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42) ## split x and y lists into training and test lists

y_train = to_categorical(y_train, num_classes=2) ## convert y training data into categorical with two classes for women and men
y_test = to_categorical(y_test, num_classes=2) ## convert y testing data into categorical with two classes for women and men

## save the four arrays so preprocessing only needs to be done once
np.save('Resources/GenderCNN/x_train.npy', x_train)
np.save('Resources/GenderCNN/x_test.npy', x_test)
np.save('Resources/GenderCNN/y_train.npy', y_train)
np.save('Resources/GenderCNN/y_test.npy', y_test)
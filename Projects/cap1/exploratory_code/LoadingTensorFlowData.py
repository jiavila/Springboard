import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from read_write_sample_imgs import open_cv10_data

data_dir = ''
categories = []

[path_cv10_data, data_cv10, labels_cv10, path_sample_imgs] = open_cv10_data()

# balance of training data
#   It's important that training data is balanced. Since our data isn't balanced, we can pass class weights to handle
#   imbalanced dataset.

# Shuffle the data
#   data needs to be shuffled so that neural net learns evenly
training_data = list(zip(data_cv10, labels_cv10))
random.shuffle(training_data)

# Pack data into variables we're going to use
X = []  # X is typically the feature set
y = []  # these are usually the labels

for features, label in training_data:
    X.append(features)
    y.append(label)

# both have to be a numpy array, and needs to be reshaped to 3d vector
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # 1 at the end to indicate greyscale


# save data...you can also use np.save('features.npy', X)
import pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)


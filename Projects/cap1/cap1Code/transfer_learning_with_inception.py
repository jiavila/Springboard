# from here: https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
# ************************************************************************* #
# Importing
# ************************************************************************* #
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import inception_v3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from deepimgbuilder import DeepImageBuilder


# ************************************************************************* #
# Loading and pre-process training data
# ************************************************************************* #
# this step was edited b/c we already have our images in a numpy array
deep_ddsm = DeepImageBuilder(path_main='D:\Documents\Springboard\ProjectData\ddsm-mammography')
deep_ddsm.get_data()

# Get a subsample of the data and store in our object
num_samples_training_dict = deep_ddsm.create_smaller_train_set(percent=10)

# Get a validation set from sampled training set. This automatically removes these samples from our training set.
num_samples_val_dict = deep_ddsm.create_val_set(percent=20)


# Prepare our data for processing
deep_ddsm.prep_data(data_choice=['training', 'validation'])

x_train = deep_ddsm.DataTrain
y_train = deep_ddsm.LabelsTrain
val_set = (deep_ddsm.DataVal, deep_ddsm.LabelsVal)



# ************************************************************************* #
# Building the required model
# ************************************************************************* #
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(5,activation='softmax')(x) #final layer with softmax activation. this is how many we categories we have.

# create a model based on our architecture
# specify the inputs
# specify the outputs
model=Model(inputs=base_model.input,outputs=preds)

# Check architecture of our model
for i,layer in enumerate(model.layers):
  print(i,layer.name)

# Make all weights untrainable except the last dense layers that we added
for layer in model.layers[:312]:
    layer.trainable = False

for layer in model.layers[312:]:
    layer.trainable = True


# ************************************************************************* #
# Load data into ImageDataGenerator, compile, and fit model
# ************************************************************************* #
datagen = ImageDataGenerator(data_format='channels_last')

# compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=2, validation_data=val_set)

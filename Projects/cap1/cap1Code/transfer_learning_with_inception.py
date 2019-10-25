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
num_samples_training_dict = deep_ddsm.create_smaller_train_set(percent=50)

# Get a validation set from sampled training set. This automatically removes these samples from our training set.
num_samples_val_dict = deep_ddsm.create_val_set(percent=20)


# Prepare our data for processing
deep_ddsm.prep_data(data_choice=['training', 'validation'])


# Put our validation set into a tuple to use later as validation during training
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
# Load data into ImageDataGenerator with parameters, compile, fit model. Do this N number of times
# ************************************************************************* #
# create dataframes that will have our information
accur_model_df = pd.DataFrame()
accur_val_df = pd.DataFrame()
num_models = 10
datagen = ImageDataGenerator(data_format='channels_last', rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, vertical_flip=True,
                             fill_mode='nearest')
for idx in range(num_models):
    print("Creating model ", str(idx+1), " of ", str(num_models))

    # compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(deep_ddsm.DataTrain)

    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(datagen.flow(deep_ddsm.DataTrain, deep_ddsm.LabelsTrain, batch_size=32),
                        steps_per_epoch=len(deep_ddsm.DataTrain) / 32, epochs=20, validation_data=val_set)

    # Save model accuracy and validation accuracy at each epoch for model_x in a DataFrames
    accur_model_df['model_' + str(idx)] = history.history['accuracy']
    accur_val_df['model_' + str(idx)] = history.history['val_accuracy']

# ************************************************************************* #
# Plot our model accuracies and validation accuracies
# ************************************************************************* #
# datagen has all the parameters of our x model
textstr = '\n'.join((
    'Rotation range: ' + str(datagen.rotation_range),
    'Zoom range: ' + str(datagen.zoom_range),
    'Width shift range: ' + str(datagen.width_shift_range),
    'Height shift range: ' + str(datagen.height_shift_range),
    'Shear range: ' + str(datagen.shear_range),
    'Horizontal flip: ' + str(datagen.horizontal_flip),
    'Vertical flip: ' + str(datagen.vertical_flip),
    'Fill mode: ' + datagen.fill_mode))
my_plot = accur_model_df.plot(title='testing')
my_plot.text(x=0.05, y=0.95, s=textstr)
plt.show()








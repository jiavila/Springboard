# This is the main script. This script loads and prepares data, imports and builds a keras CNN architecture, and trains
# N number of models based on the imported keras CNN. Each of the models uses the same training data, but
# ImageDataGenerator applies arbitrary transformations to the source training images. This is to evaluate how well the
# model can generalize (i.e., evaluate its robustness to transformations in image data).
# ************************************************************************* #
# Importing
# ************************************************************************* #
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import inception_v3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from deepimgbuilder import DeepImageBuilder
import config as cfg


# ************************************************************************* #
# Load and pre-process data with deepimgbuilder class
# ************************************************************************* #
# this step was edited b/c we already have our images in a numpy array
deep_ddsm = DeepImageBuilder(paths_dict=cfg.paths_dict)
deep_ddsm.get_data()

# Get a subsample of the data and store in our object
num_samples_training_dict = deep_ddsm.create_smaller_train_set(percent=cfg.percent_train_set)

# Get a validation set from sampled training set. This automatically removes these samples from our training set.
if cfg.create_val_set_bool:
    num_samples_val_dict = deep_ddsm.create_val_set(percent=cfg.percent_val_set)


# Prepare our data for processing
deep_ddsm.prep_data(data_choice=['training', 'validation'])


# ************************************************************************* #
# Building the required model. Change later to import different learning model from config file
# ************************************************************************* #
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(cfg.num_classes,activation='softmax')(x) #final layer with softmax activation. this is how many we categories we have.

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
# create dataframes that will hold accuracy and validation accuracy information from each model that's built
accur_model_df = pd.DataFrame()
accur_val_df = pd.DataFrame()
num_models = cfg.num_models
datagen = ImageDataGenerator(data_format=cfg.data_format, rotation_range=cfg.rotation_range, zoom_range=cfg.zoom_range,
                             width_shift_range=cfg.width_shift_range, height_shift_range=cfg.height_shift_range,
                             shear_range=cfg.shear_range, horizontal_flip=cfg.horizontal_flip,
                             vertical_flip=cfg.vertical_flip, fill_mode=cfg.fill_mode)
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
    history = model.fit_generator(datagen.flow(deep_ddsm.DataTrain, deep_ddsm.LabelsTrain, batch_size=cfg.batch_size),
                                  steps_per_epoch=len(deep_ddsm.DataTrain) / cfg.batch_size, epochs=cfg.epochs,
                                  validation_data=(deep_ddsm.DataVal, deep_ddsm.LabelsVal))

    # Save model accuracy and validation accuracy at each epoch for model_x in a DataFrames
    accur_model_df['model_' + str(idx)] = history.history['accuracy']
    accur_val_df['model_' + str(idx)] = history.history['val_accuracy']

# ************************************************************************* #
# Plot model accuracies
# ************************************************************************* #
# FUTURE: change below to ("{key}".format(**dic)) once config is turned into a json file
textstr = ''.join((
    'Rotation range = ' + str(cfg.rotation_range),
    ', Zoom range = ' + str(cfg.zoom_range),
    ', Width shift range: ' + str(cfg.width_shift_range),
    ', Height shift range: ' + str(cfg.height_shift_range),
    ', Shear range: ' + str(cfg.shear_range),
    ', Horizontal flip: ' + str(cfg.horizontal_flip),
    ', Vertical flip: ' + str(cfg.vertical_flip),
    ', Fill mode: ' + cfg.fill_mode))
accur_model_df.plot(title='Model self accuracy per epoch' + '\n' + textstr)
accur_val_df.plot(title='Validation accuracy of each model per epoch' + '\n' + textstr)
plt.show()

# This is the configuration file for MAIN_xf_learning.py.
# Future: change into .json file to import everything as a dictionary
# ************************************************************************* #
# loading and pre-process training data params
# ************************************************************************* #
percent_train_set = 5  # create a smaller training set by specifying percentage of original to use. Set to 100 for all
create_val_set_bool = True  # set True to make a validation set from sampled training set. Sampled data from training
#                                 set is removed from training set.
percent_val_set = 20        # percent of training set convert into validation.
paths_dict = {
    'path_main': 'C:\\Users\\jesus\\Documents\\Springboard\\project_data\\ddsm-mammography',
    'path_data_train': 'cv10_data\\cv10_data.npy',
    'path_labels_train': 'cv10_labels.npy',
    'path_data_test': 'test10_data\\test10_data.npy',
    'path_labels_test': 'test10_labels.npy',
    'path_data_val': '',
    'path_labels_val': ''}


# ************************************************************************* #
# Model params
# ************************************************************************* #
# All models will use the same training and validation sources, but there
# will  be slight variations in each model's training set images. Random
# transformations are applied by ImageDataGenerator object specified by
# ImageDataGenerator params.
num_models = 1      # number of models to build with arbitrary translations
#                       in training set images
num_classes = 5     # number of categories to classify
#                       base_model = inception_v3.InceptionV3(weights='
#                       imagenet', include_top=False) # future: the keras
#                        pre-built model
epochs = 3          # For each model, number of times to fit the model with
#                       the entire training set. Do this N times because
#                       stochastic gradient descent is an iterative process
batch_size = 32     # For each epoch, split data into N samples and update
#                       CNN weights after each batch


# ************************************************************************* #
# ImageDataGenerator params (params from https://keras.io/preprocessing/image/, ImageDataGenerator class)
# ************************************************************************* #
data_format = 'channels_last'   # "channels_last" mode means that the images should have shape (samples, height, width, channels)
rotation_range = 20             # Int. Degree range for random rotations.
zoom_range = 0.15               # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = # [1-zoom_range, 1+zoom_range]
width_shift_range = 0.2         # fraction of total width if < 1. all possible floats in range [-x, x)
height_shift_range = 0.2        # fraction of total height if < 1.  all possible floats in range [-x, x)
shear_range = 0.15              # Shear Intensity (Shear angle in counter-clockwise direction in degrees)
horizontal_flip = True          # arbitrarily flips an image when True
vertical_flip = True            # arbitrarily flips an image when True
fill_mode = 'nearest'           # Points outside the boundaries of the input are filled according to the given mode


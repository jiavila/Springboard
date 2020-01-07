import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd
from skimage import exposure as xp

class DeepImageBuilder:
    '''
        A deep learning class for transfer learning using pre-built keras models that were built from rgb images. This
        class helps loading and preparing data in the right format for keras models. For example, it transforms data
        from grayscale images to rgb.
    '''

    # A dictionary to match class attributes to data name variables. Primarily used in for loops within class methods.
    DATA_ATTRIBUTE_NAMES_DICT = {
        'path_data_train': 'DataTrain',
        'path_labels_train': 'LabelsTrain',
        'path_data_test': 'DataTest',
        'path_labels_test': 'LabelsTest',
        'path_data_val': 'DataVal',
        'path_labels_val': 'LabelsVal'}

    def __init__(self, paths_dict):
        self.PathsDict = self.set_paths(paths_dict, nargout=1)  # set the dictionary of paths.
        self.PathCurrent = os.path.dirname(os.path.realpath(__file__))  # path where this file is opened
        self.PathSampleImages = os.path.join(self.PathCurrent, '../sample_imgs')  # path for saved sample class images
        self.DataTrain = np.empty(shape=(100, 10, 10, 1))  # initialize with empty numpy array
        self.DataTest = np.empty(shape=(10, 10, 10, 1))  # initialize with empty numpy array
        self.DataVal = np.empty(shape=(10, 10, 10, 1))  # validation data. can be created with self.create_val_set()
        self.LabelsTrain = np.empty(shape=self.DataTrain.shape[0])
        self.LabelsTest = np.empty(shape=self.DataTest.shape[0])
        self.LabelsVal = np.empty(shape=self.DataVal.shape[0])
        self.EncoderTrain = LabelEncoder()
        self.EncoderTest = LabelEncoder()
        self.EncoderVal = LabelEncoder()

    def set_paths(self, paths_dict, nargout=0):
        '''
        Sets a dictionary that contains the absolute path of the main directory and its relative paths to the training,
        validation, and test sets (data & labels).
        :param paths_dict: Dict.
            path_main: the absolute path of the main directory that contains all the data
            path_data_train: relative path (from path_main) to the training images. Format should be numpy array (.npy).
            path_labels_train: relative path to the training labels. Format should be numpy array (.npy)
            path_data_test, path_labels_test, path_data_val, path_labels_val: similar to training paths. If not
                available, set to ''
            Example:
                paths_dict = {
                    'path_main': 'C:\\Users\\jesus\\Documents\\Springboard\\project_data\\ddsm-mammography',
                    'path_data_train': 'cv10_data\\cv10_data.npy',
                    'path_labels_train': 'cv10_labels.npy',
                    'path_data_test': 'test10_data\\test10_data.npy',
                    'path_labels_test': 'test10_labels.npy',
                    'path_data_val': '',
                    'path_labels_val': ''}
        :param nargout: Int. Output paths_dict if nargout = 1 (after checking key names). Do a self update
            (self.PathsDict) if 0.
        :return:
        '''
        args = ('path_main',) + tuple(self.DATA_ATTRIBUTE_NAMES_DICT.keys())
        for key, value in paths_dict.items():
            if not(key in args):
                raise ValueError(key, 'Not recognized. path keys must be one of the following:', args)
        if nargout == 0:
            self.PathsDict = paths_dict
        elif nargout == 1:
            return paths_dict
        else:
            ValueError(nargout, 'must be int 0 or 1')

    def get_data(self):
        data_attribute_names_dict = self.DATA_ATTRIBUTE_NAMES_DICT
        '''
        Loads image data and labels for training, validation, or test data. Only loads data if the file path for each 
        file isn't empty.
        :return:
        '''
        for key, file_path in self.PathsDict.items():
            if (file_path != '') & (type(file_path) == str) & (key != 'path_main'):
                exec("self." + str(data_attribute_names_dict[key]) +
                     r" = np.load(self.PathsDict['path_main'] + '\\'  + file_path)")

    def show_sample_class_images(self, num_classes, display_images=True, save_images=False):
        '''
        FUTURE: create a method that shows a sample image from each class/category.
        :param num_classes:
        :param display_images:
        :param save_images:
        :return:
        '''

        # *************************************************************#
        # Get the indeces for the differen types of classes. Create static method from "sample for each type of class"
        # in static method get_sample
        if save_images:
            print('Sample class images will be saved in: ' + self.PathSampleImages)

    def check_data_choice(self, data_choice: list = None):
        # *************************************************************#
        # Check data_choice type if all the strings in data_choice are correct
        if type(data_choice) != list:
            raise TypeError(
                "data_choice must be a list. List must contain 'training', 'test', and/or 'validation'")

        for entry in data_choice:
            if not (entry in ['training', 'test', 'validation']):
                raise ValueError(
                    entry + "not recognized. data_choice list can only contain 'training', 'test', and/or "
                            "'validation'.")

        # *************************************************************#
        # Get the corresponding data and training labels
        data_list = []
        labels_list = []
        data_choice_tracker = []
        if 'training' in data_choice:
            data_list.append(self.DataTrain)
            labels_list.append(self.LabelsTrain)
            data_choice_tracker.append('Train')
            print("Selected training set to prep.")
        if 'test' in data_choice:
            data_list.append(self.DataTest)
            labels_list.append(self.LabelsTest)
            data_choice_tracker.append('Test')
            print("Selected test set to prep.")
        if 'validation' in data_choice:
            data_list.append(self.DataVal)
            labels_list.append(self.LabelsVal)
            data_choice_tracker.append('Val')
            print("Selected validation set to prep.")

        return data_list, labels_list, data_choice_tracker

    def prep_data(self, data_choice):
        '''
        Prepare data by converting images from gray scale NxNx1 to rgb NxNx3. This is done because the imported keras
        models were trained with rgb images. In addition, one-hot encode labels if it's necessary.
        Future: generalize storing data to self like get_data() method by using DATA_ATTRIBUTE_NAMES_DICT
        :param data_choice: a list that contains 'training', 'test', and/or 'validation'. Data preparation will apply
                            to data stored in corresponding attributes.
        :return:
        '''

        data_list, labels_list, data_choice_tracker = \
            self.check_data_choice(data_choice=data_choice)

        # *************************************************************#
        # Loop through data lists and prepare data if it's needed
        for idx, choice in enumerate(data_choice_tracker):

            # *************************************************************#
            # Assign current variables
            data = data_list[idx]
            labels = labels_list[idx]
            suffix = choice

            # *************************************************************#
            # Convert images from gray scale NxNx1 to rgb NxNx3 (since our imported nodes were built from rgb images)
            if data.shape[-1] == 1:
                print("Converting images from grayscale NxNx1 to rgb NxNx3")
                data = np.repeat(data, 3, -1)
                print("...New shape of data: ", data.shape)

            # *************************************************************#
            # converting our training labels to an NxM matrix

            # convert label strings to numerical
            if type(labels[0]) == str:
                print("Converting labels from strings to numerical. Outputting encoder to ")
                encoder = LabelEncoder()
                encoder.fit(labels)
                labels = encoder.transform(labels)
                exec("self.Encoder" + suffix + " = encoder")

            # Convert integers to dummy variables (i.e., one-hot encoded)
            if len(labels.shape) == 1:
                print("Converting labels to categorical (one-hot encoded)")
                labels = np_utils.to_categorical(labels)

            # Store arguments into respective properties
            self.store_loop_data(data, labels, suffix)

    def adjust_exposure(self, data_choice):

        # Check data choice
        data_list, labels_list, data_choice_tracker = \
            self.check_data_choice(data_choice=data_choice)

        # *************************************************************#
        # Loop through data lists and adjust exposure
        for idx, choice in enumerate(data_choice_tracker):
            # *************************************************************#
            # Assign current variables
            data = data_list[idx]
            labels = labels_list[idx]
            suffix = choice

            # 

            # Store arguments into respective properties
            self.store_loop_data(data, labels, suffix)

    @staticmethod
    def store_loop_data(data, labels, suffix):
        """
        Store arguments into respective properties.

        This is typically run inside a loop, where each argument is the
        current value.

        :param data:
        :param labels:
        :param suffix:
        :return:
        """

        exec("self.Data" + suffix + " = data")
        exec("self.Labels" + suffix + " = labels")
        print(
            "Storing data in self.Data" + suffix + " and labels in self.Labels" + suffix)

    def create_smaller_train_set(self, percent):
        """
        Subsample the training set of our object and replace the results in self.DataTrain and self.LabelsTrain
        :param percent: percent of the dataset we want to sample
        :return: num_samples_training_dict, a dictionary containing the number of samples for each class
        """
        [self.DataTrain, self.LabelsTrain, num_samples_training_dict] = DeepImageBuilder.get_sample(
            data=self.DataTrain,
            labels=self.LabelsTrain, percent=percent,
            remove_samples=False)
        return num_samples_training_dict

    def create_val_set(self, percent):
        """
        Creates a validation set (self.DataVal, self.LabelsVal) from the training set (self.DataTrain, self.LabelsTrain)
        This automatically removes these samples from our training set.
        :param percent: percent of training data to convert to validation set
        :return: num_samples_validation_dict, a dictionary containing the number of samples for each class
        """
        [self.DataVal, self.LabelsVal, num_samples_validation_dict, self.DataTrain, self.LabelsTrain] = \
            DeepImageBuilder.get_sample(data=self.DataTrain, labels=self.LabelsTrain, percent=percent,
                                        remove_samples=True)
        return num_samples_validation_dict

    @staticmethod
    def get_sample(data, labels, percent, remove_samples):
        '''
        Create a subsmaple of our data. This should be done before method DeepImageBuilder.prep_data() is called.
        :param data: DeepImageBuilder.Data
        :param labels: DeepImageBuilder.Labels, should be an Nx1 numpy array
        :param percent: int, the sample size percent
        :param remove_samples: boolean, set to True to remove the samples that were sampled from data
        :return: [data_sampled, labels_sampled, num_samples_dict, data_samples_removed, labels_samples_removed] if
                    remove_samples = True, else return
                    [data_sampled, labels_sampled, num_samples_dict]
        '''

        # *************************************************************#
        # pre-allocate sampled numpy arrays
        num_samples_dict = pd.Series(labels).value_counts()  # count number of samples of the sampled array
        num_samples_dict = dict(np.ceil(num_samples_dict*percent/100).astype('int64'))  # have at least one sample
        data_sampled = np.empty(shape=(sum(num_samples_dict.values()),) + data.shape[1:], dtype=data.dtype)
        labels_sampled = np.empty(shape=data_sampled.shape[0], dtype=labels.dtype)
        idx_tracker = 0  # keep track of the starting index of our sample
        idx_samples_all = []  # an array to keep track of all the indexes we sampled

        # *************************************************************#
        # sample for each type of class
        for label, num_samples in num_samples_dict.items():
            # get the indexes of labels that match our current label
            idx_labels = np.where(labels == label)[0]  # [0] because it's a tuple with our array

            # sample the dictionary
            idx_labels_sampled = np.random.choice(idx_labels, size=num_samples, replace=False)
            data_sampled[idx_tracker:idx_tracker + num_samples, :, :, :] = data[idx_labels_sampled, :, :, :]
            labels_sampled[idx_tracker:idx_tracker + num_samples] = labels[idx_labels_sampled]
            idx_tracker += num_samples

            idx_samples_all.extend(idx_labels_sampled)

        # *************************************************************#
        # Randomly shuffle the data and labels
        shuffler = np.arange(labels_sampled.shape[0])  # create an indexed array
        np.random.shuffle(shuffler)  # shuffle our array
        data_sampled = data_sampled[shuffler, :, :, :]
        labels_sampled = labels_sampled[shuffler]

        if remove_samples:
            # create a mask to remove sampled data and return
            mask = np.ones(len(labels), dtype=bool)
            mask[tuple([idx_samples_all])] = False
            data_samples_removed = data[mask, :, :, :]
            labels_samples_removed = labels[mask]
            return [data_sampled, labels_sampled, num_samples_dict, data_samples_removed, labels_samples_removed]
        else:
            return [data_sampled, labels_sampled, num_samples_dict]

    @staticmethod
    def plot_history(history, datagen):
        """
        Future: if it becomes necessary, create a static plotting method for the class
        :param history: keras.callbacks.callbacks.History object, output from keras.models.Model.fit_generator()
        :param datagen: keras.preprocessing.image.ImageDataGenerator, the generator object that was used when fitting
                        Model.fit_gerator(). Use this to get parameters.
        :return:
        """


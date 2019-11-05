import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd

class DeepImageBuilder:
    '''
    A deep learning class for transfer learning using pre-built keras models from rgb images.
    '''
    def __init__(self, path_main):
        self.PathMain = path_main  # the path to the directory where our data are stored
        self.DataTrain = np.empty(shape=(100, 10, 10, 1))  # initialize with empty numpy array
        self.DataTest = np.empty(shape=(10, 10, 10, 1))  # initialize with empty numpy array
        self.DataVal = np.empty(shape=(10, 10, 10, 1))  # validation data. this can be created with get_sample method
        self.LabelsTrain = np.empty(shape=self.DataTrain.shape[0])
        self.LabelsTest = np.empty(shape=self.DataTest.shape[0])
        self.LabelsVal = np.empty(shape=self.DataVal.shape[0])
        self.PathCurrent = ''
        self.PathSampleImages = ''
        self.EncoderTrain = LabelEncoder()
        self.EncoderTest = LabelEncoder()
        self.EncoderVal = LabelEncoder()

    def set_paths(self, path_main):
        if path_main:  # if the variable is empty
            self.PathMain = 'D:\Documents\Springboard\ProjectData\ddsm-mammography'  # default directory
        else:
            self.PathMain = path_main

    def get_data(self):
        '''
        Fix later. gets the data
        :return:
        '''
        self.DataTrain = np.load(self.PathMain + '\\cv10_data\\cv10_data.npy')
        self.LabelsTrain = np.load(os.path.join(self.PathMain, 'cv10_labels.npy'))
        self.DataTest = np.load(self.PathMain + '\\test10_data\\test10_data.npy')
        self.LabelsTest = np.load(os.path.join(self.PathMain, 'test10_labels.npy'))
        self.PathCurrent = os.path.dirname(os.path.realpath(__file__))
        self.PathSampleImages = os.path.join(self.PathCurrent, '../sample_imgs')
        print('Current path: ' + self.PathCurrent)
        print('Sample images path: ' + self.PathSampleImages)

    def prep_data(self, data_choice):
        '''
        Prepare data by converting images from gray scale NxNx1 to rgb NxNx3. This is done because the imported keras
        models were trained with rgb images.
        :param data_choice: a list that contains 'training', 'test', and/or 'validation'. Data preparation will apply
                            to data stored in corresponding attributes.
        :return:
        '''

        # *************************************************************#
        # Check data_choice type if all the strings in data_choice are correct
        if type(data_choice) != list:
            raise TypeError("data_choice must be a list. List must contain 'training', 'test', and/or 'validation'")

        for entry in data_choice:
            if not(entry in ['training', 'test', 'validation']):
                raise ValueError(entry + "not recognized. data_choice list can only contain 'training', 'test', and/or "
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

            # *************************************************************#
            # Store current variables in respective attributes
            exec("self.Data" + suffix + " = data")
            exec("self.Labels" + suffix + " = labels")
            print("Storing prepped data in self.Data" + suffix + " and labels in self.Labels" + suffix)

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

        :param history: keras.callbacks.callbacks.History object, output from keras.models.Model.fit_generator()
        :param datagen: keras.preprocessing.image.ImageDataGenerator, the generator object that was used when fitting
                        Model.fit_gerator(). Use this to get parameters.
        :return:
        """


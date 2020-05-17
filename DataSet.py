import numpy as np
from classes.AudioRecording import AudioRecording


class DataSet:

    def __init__(self, data_array, label_array, is_raw_data=False, sampling_rate=None):

        if not isinstance(is_raw_data, bool):
            raise TypeError("The argument 'is_raw_data' must be a boolean")
        if not isinstance(data_array, np.ndarray):
            raise TypeError("The data set must be a numpy array")
        if not isinstance(label_array, np.ndarray):
            raise TypeError("The set of labels must be a numpy array")
        if len(np.shape(label_array)) != 2:
            raise ValueError("The set of labels must have exactly two dimensions")
        if len(data_array) != len(label_array):
            raise ValueError("The sizes of the sets of data and labels do not match")
        if len(data_array) == 0:
            raise ValueError("The data sets provided are empty")
        if len(label_array[0]) < 1:
            raise ValueError("The number of labels in the data set must be at least one")

        if is_raw_data:
            if sampling_rate is None:
                raise TypeError("The sampling rate must be provided for raw data sets")
            print("building data set")
            self.build_from_raw_data(data_array, label_array, sampling_rate=sampling_rate)
            print("data set built")
        else:
            if len(np.shape(data_array)) != 2:
                raise ValueError("The data set must have exactly two dimensions")
            if len(data_array[0]) < 1:
                raise ValueError("The number of features in the data set must be at least one")
            self.data = data_array
            self.labels = label_array
            self.raw_data = None
        (self.N, self.F) = np.shape(self.data)
        (self.N, self.C) = np.shape(self.labels)
        self.is_normalized = False
        self.mean = np.mean(self.data, axis=0)
        self.standard_deviation = np.std(self.data, axis=0)

    def build_from_raw_data(self, raw_data_array, label_array, sampling_rate=2000):
        """
        Builds the data set from an array of audio recordings
        :param raw_data_array: (float array) array of audio recordings
        :param label_array: (int NxC array) array of the labels
        :param sampling_rate: (float) sampling rate of the recordings
        :return: None
        """

        if not isinstance(raw_data_array, np.ndarray):
            raise TypeError("The data set must be a numpy array")
        if not isinstance(label_array, np.ndarray):
            raise TypeError("The set of labels must be a numpy array")
        if len(np.shape(label_array)) != 2:
            raise ValueError("The set of labels must have exactly two dimensions")
        if len(raw_data_array) != len(label_array):
            raise ValueError("The sizes of the sets of data and labels do not match")
        if len(raw_data_array) == 0:
            raise ValueError("The data sets provided are empty")
        if len(label_array[0]) < 1:
            raise ValueError("The number of labels in the data set must be at least one")
        if not (isinstance(sampling_rate, float) or isinstance(sampling_rate, int)):
            raise TypeError("The sampling rate must be a positive float or integer")
        if min([len(x) for x in raw_data_array]) < 3:
            raise ValueError("Each recording in the data set must have a length of a least three")

        data = []
        raw_data = []
        labels = []

        for i in range(len(raw_data_array)):
            raw_sample = AudioRecording(raw_data_array[i], sampling_rate)
            features = raw_sample.extract_features()
            raw_sample.zero_padding()
            for j in range(len(features)):
                if j % 3 == 0:
                    data.append(features[j])
                    raw_data.append(raw_sample.segments[j])
                    labels.append(label_array[i])

        self.data = np.array(data)
        self.raw_data = np.array(raw_data)
        self.labels = np.array(labels)

    def save(self, data_file_name, labels_file_name):
        """
        Saves the data set into two npy files
        :param data_file_name: (sting) name of the file of the data set
        :param labels_file_name: (string) name of the file of the set of labels
        :return: None
        """

        if not isinstance(data_file_name, str) or not isinstance(labels_file_name, str):
            raise TypeError("The file names must be strings")
        if data_file_name[-4:] != ".npy" or labels_file_name[-4:] != ".npy":
            raise ValueError("The file names must end with '.npy'")
        np.save(data_file_name, self.data)
        np.save(labels_file_name, self.labels)

    def load(self, data_file_name, labels_file_name):
        """
        Loads the data set from two npy files
        :param data_file_name: (sting) name of the file of the data set
        :param labels_file_name: (string) name of the file of the set of labels
        :return: None
        """

        if not isinstance(data_file_name, str) or not isinstance(labels_file_name, str):
            raise TypeError("The file names must be strings")
        if data_file_name[-4:] != ".npy" or labels_file_name[-4:] != ".npy":
            raise ValueError("The file names must end with '.npy'")
        self.data = np.load(data_file_name)
        self.labels = np.load(labels_file_name)

    def shuffle(self):
        """
        Randomly shuffles the data set
        :return: None
        """

        np.random.seed(521)
        rand_indices = np.arange(self.N)
        np.random.shuffle(rand_indices)
        self.data = self.data[rand_indices]
        self.labels = self.labels[rand_indices]

    def shape(self):
        """
        Returns the shape of the data set
        :return: N (int), number of samples; F (int), number of features; C (int), number of classes
        """
        return self.N, self.F, self.C

    def normalize(self):
        """
        Normalizes the features present in the data set and sets "self.is_normalized" to True.
        :return: None
        """
        if np.min(self.standard_deviation) == 0:
            raise ValueError("The data set has a constant feature")
        one_vector = np.ones(self.N)
        self.data = (self.data - np.outer(one_vector, self.mean))/np.outer(one_vector, self.standard_deviation)
        self.is_normalized = True

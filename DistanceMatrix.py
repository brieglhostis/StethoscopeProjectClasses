import numpy as np


class DistanceMatrix:

    distance_types = ["euclidian", "jaccard", "cosine"]

    def __init__(self, training_data, data, distance_type):
        if not isinstance(training_data, np.ndarray):
            raise TypeError("The training data set must be a numpy array")
        if not isinstance(training_data, np.ndarray):
            raise TypeError("The training data set must be a numpy array")
        if len(np.shape(training_data)) != 2:
            raise ValueError("The training data set must have exactly two dimensions")
        if len(training_data) == 0:
            raise ValueError("The training data set must not be empty")
        if len(training_data[0]) == 0:
            raise ValueError("The training data set must have at least one feature")
        if not isinstance(data, np.ndarray):
            raise TypeError("The tested data must be a numpy array")
        if len(np.shape(data)) != 2:
            raise ValueError("The tested data must have exactly two dimensions")
        if len(data) == 0:
            raise ValueError("The tested data must not be empty")
        if len(training_data[0]) != len(data[0]):
            raise ValueError("The tested data and the training data set must have the same number of features")

        if distance_type in self.distance_types:
            self.distance_type = distance_type
            self.training_data = training_data
            self.data = data
            self.distance_matrix = np.zeros((len(self.data), len(self.training_data)))
            if distance_type == "euclidian":
                self.euclidian_distances()
            if distance_type == "jaccard":
                self.jaccard_distances()
            if distance_type == "cosine":
                self.cosine_distances()
        else:
            raise ValueError("Unknown distance")

    def euclidian_distances(self):
        """
        Computes the matrix of the Euclidian distances between the training data and the data to evaluate
        :return: None
        """
        n_train = len(self.training_data)
        n_valid = len(self.data)
        for i in range(n_valid):
            self.distance_matrix[i] = np.sqrt(
                np.sum(np.power(np.outer(np.ones(n_train), self.data[i]) - self.training_data, 2), axis=1))

    def jaccard_distances(self):
        """
        Computes the matrix of the Jaccard distances between the training data and the data to evaluate
        :return: None
        """
        n_train = len(self.training_data)
        n_valid = len(self.data)
        one_vector = np.ones(n_train)
        min_training_data = np.amin(self.training_data) * one_vector
        for i in range(n_valid):
            denominator = np.sum(np.maximum(np.outer(np.ones(n_train), self.data[i]), self.training_data), axis=1)\
                          - min_training_data
            if np.min(np.abs(denominator)) == 0:
                for j in range(n_train):
                    if denominator[j] == 0:
                        denominator[j] = 1e-12
            self.distance_matrix[i] = one_vector - (np.sum(np.minimum(np.outer(np.ones(n_train), self.data[i]),
                                                                      self.training_data), axis=1)
                                                    - min_training_data) / denominator

    def cosine_distances(self):
        """
        Computes the matrix of the Cosine distances between the training data and the data to evaluate
        :return: None
        """
        n_train = len(self.training_data)
        n_valid = len(self.data)
        scalar_products = np.zeros((n_valid, n_train))
        for i in range(n_valid):
            scalar_products[i] = np.sum(np.outer(np.ones(n_train), self.data[i]) * self.training_data, axis=1)
        train_norm = np.outer(np.ones(n_valid), np.sqrt(np.sum(np.power(self.training_data, 2), axis=1)))
        valid_norm = np.outer(np.sqrt(np.sum(np.power(self.data, 2), axis=1)), np.ones(n_train))
        similarity = scalar_products/(train_norm*valid_norm)
        for i in range(n_valid):
            for j in range(n_train):
                if similarity[i][j] <= 0:
                    similarity[i][j] = 1e-12
                elif similarity[i][j] >= 1:
                    similarity[i][j] = 1 - 1e-12
        self.distance_matrix = np.arccos(similarity)/np.pi

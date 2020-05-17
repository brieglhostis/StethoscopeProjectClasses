import numpy as np
from classes.DistanceMatrix import DistanceMatrix
from classes.DataSet import DataSet
from classes.ModelAssessment import ModelAssessment


class KNNModel:

    features_names = [
        "Mean",
        "Median",
        "Standard deviation",
        "25th percentile",
        "75th percentile",
        "Mean Absolute Deviation",
        "Inter Quartile Range",
        "Skewness",
        "Kurtosis",
        "Shannon's entropy",
        "Spectral entropy",
        "Maximum frequency",
        "Maximum magnitude",
        "Ratio of signal energy",
        "MFCC 1",
        "MFCC 2",
        "MFCC 3",
        "MFCC 4",
        "MFCC 5",
        "MFCC 6",
        "MFCC 7",
        "MFCC 8",
        "MFCC 9",
        "MFCC 10",
        "MFCC 11",
        "MFCC 12",
        "MFCC 13"
    ]
    F = len(features_names)
    distance_types = ["euclidian", "jaccard", "cosine"]

    def __init__(self, training_data_set, K=1, distance_type="jaccard", features_list=None):
        if not isinstance(training_data_set, DataSet):
            raise TypeError("The training data set must be an instance of the class DataSet")
        if not isinstance(K, int):
            raise TypeError("The number of neighbors must be a positive integer")
        if K < 1 or K > training_data_set.N-1:
            raise ValueError("The number of neighbors must be a positive integer that is less than the number of"
                             " samples in the training data set minus one")
        if not (distance_type in self.distance_types):
            raise ValueError("Unknown distance type")
        if training_data_set.F == self.F:
            if features_list is None:
                features_list = self.features_names
            else:
                if not isinstance(features_list, list):
                    raise TypeError("The argument 'features_list' should be a list")
                for name in features_list:
                    if name not in self.features_names:
                        raise ValueError(f"The feature '{name}' is not a valid feature")
            self.features_list = features_list
            self.mapping_features = [features_list.index(self.features_names[i]) for i in range(self.F) if
                                     self.features_names[i] in features_list]
        else:
            self.mapping_features = np.arange(training_data_set.F)
            self.features_list = None

        training_data = training_data_set.data[:, self.mapping_features]
        self.training_data_set = DataSet(training_data, training_data_set.labels)
        self.K = K
        self.evaluation = None
        self.distance_type = distance_type
        self.predictions, self.nearest_indices = None, None

    def evaluate_data_set(self, data, normalize=True, return_indices=False):
        """
        Evaluates the predictions of the KNN model for a data set
        :param data: (float NxF array) data set to be evaluated
        :param normalize: (bool) if True then both the data set to predict and the training data set will be normalized
        :param return_indices: (bool) if True then the method will return the matrix of the indices of nearest neighbors
        :return: predictions (int NxC array) array of the predictions of the rows of data  for each class in the
        training data set
        """

        if not isinstance(normalize, bool):
            raise TypeError("The argument 'normalize' must be a boolean")
        if not isinstance(data, np.ndarray):
            raise TypeError("The data set must be a numpy array")
        if len(np.shape(data)) != 2:
            raise ValueError("The data set must have exactly two dimensions")
        if len(data) == 0:
            raise ValueError("The data set provided is empty")
        if len(data[0]) < 1:
            raise ValueError("The number of features in the data set must be at least one")
        if len(data[0]) == self.F:
            data = data[:, self.mapping_features]
        elif len(data[0]) != self.training_data_set.F:
            raise ValueError("The number of features in the data set must match the number of features in the training "
                             "data set")

        def get_weights(k_nearest_distances_):
            """
            Computes the weights for the k nearest neighbors in the KNN model (proportional to the inverse of the
            distance)
            :param k_nearest_distances_: (float Kx1 array) array of the distances of the K nearest neighbors
            :return: (float Kx1 array) array of the weights of the K nearest neighbors
            """
            (N_data, k) = np.shape(k_nearest_distances_)
            if k == 1:
                return np.ones((N_data, k))
            for index_ in range(N_data):
                if np.max(k_nearest_distances_[index_]) == 0:
                    k_nearest_distances_[index_] = np.ones(k)
            return 1 / (k_nearest_distances_ * np.outer(np.sum(1 / k_nearest_distances_, axis=1), np.ones(k)))

        (N, F) = np.shape(data)  # F = 27
        c = self.training_data_set.C  # 2
        data_before_normalization = np.copy(data)

        data_set = DataSet(self.training_data_set.data, self.training_data_set.labels)
        if normalize and not data_set.is_normalized and np.min(data_set.standard_deviation) != 0:
            data_set.normalize()
            mean = data_set.mean
            standard_deviation = data_set.standard_deviation
            one_vector = np.ones(N)
            data = (data - np.outer(one_vector, mean)) / np.outer(one_vector, standard_deviation)

        votes = np.zeros((N, c))
        predictions = np.zeros((N, c))

        k_nearest_distances = np.zeros((N, self.K))
        k_nearest_labels = np.zeros((N, self.K, c))
        nearest_indices = np.zeros((N, self.K))
        maximum_size_of_evaluation = 100
        i = 0
        while i < N:
            sub_set = np.arange(i, min(i+maximum_size_of_evaluation, N))
            n = len(data[sub_set])
            if n > 0:
                dm = DistanceMatrix(data_set.data, data[sub_set], self.distance_type)
                distances = dm.distance_matrix
                ordering = np.argsort(distances, axis=1)
                k_nearest_indices = ordering[:, :self.K + 1]
                for index in sub_set:
                    index_inter = index - np.min(sub_set)
                    index_to_remove = np.argwhere(np.all(self.training_data_set.data ==
                                                         data_before_normalization[index], axis=1))
                    if len(index_to_remove) > 0:
                        index_to_remove = index_to_remove[:, 0]
                        indices_left = np.array(
                            [i for i in range(self.K + 1) if k_nearest_indices[index_inter][i] not in
                             index_to_remove][:self.K])
                        k_nearest_indices_index = k_nearest_indices[index_inter][indices_left]
                    else:
                        k_nearest_indices_index = k_nearest_indices[index_inter][:-1]
                    nearest_indices[index] = k_nearest_indices_index
                    k_nearest_distances[index] = distances[index_inter][
                        k_nearest_indices_index]  # Get the K nearest neighbour's distances
                    k_nearest_labels[index, :, :] = data_set.labels[k_nearest_indices_index]
                i += maximum_size_of_evaluation
            else:
                break

        weights = get_weights(k_nearest_distances)  # Get the weights from the  distances

        for index in range(N):
            votes[index] = np.sum(np.outer(weights[index], np.ones(c)) * k_nearest_labels[index],
                                  axis=0)  # Compute votes for each class
            predictions[index][np.argmax(votes[index])] = 1

        if return_indices:
            return predictions, nearest_indices.astype(np.int)
        else:
            return predictions

    def self_evaluation(self, normalize=True):
        """
        Applies the "evaluate_data_set_method" on the training data set and sets self.predictions and
        self.nearest_indices.
        Note that for a sample in the training database, this sample will note be taken into account as its neighbor
        :param normalize: (bool) if True then both the data set to predict and the training data set will be normalized
        :return: None
        """
        self.predictions, self.nearest_indices = self.evaluate_data_set(self.training_data_set.data,
                                                                        normalize=normalize, return_indices=True)

    def assess_model(self, print_evaluation=False, cross_validation=False, normalize=True):
        """
        Assesses the model based on the criteria defined in the 2016 PhysioNet Challenge:
        https://physionet.org/content/challenge-2016/1.0.0/
        :param print_evaluation: (bool) if True then the criteria will be printed in the console
        :param cross_validation: (bool) if True then the assessment will use a 5-fold cross-validation method, else,
        it will simply set 80% of the data set as training set and the rest as testing set
        :param normalize: (bool) if True then both the data set to predict and the training data set will be normalized
        :return:
        """
        if not isinstance(print_evaluation, bool):
            raise TypeError("The argument 'print_evaluation' must be a boolean")
        if not isinstance(cross_validation, bool):
            raise TypeError("The argument 'cross_valuation' must be a boolean")
        if not isinstance(normalize, bool):
            raise TypeError("The argument 'normalize' must be a boolean")

        data_set = DataSet(self.training_data_set.data, self.training_data_set.labels)

        data_set.shuffle()
        size_of_validation_set = int(0.2 * data_set.N)

        if cross_validation:
            evaluations = []
            for i in range(5):
                training_data = np.concatenate((data_set.data[:i*size_of_validation_set],
                                                data_set.data[(i+1)*size_of_validation_set:]))
                training_labels = np.concatenate((data_set.labels[:i*size_of_validation_set],
                                                  data_set.labels[(i+1)*size_of_validation_set:]))

                validation_data = data_set.data[i*size_of_validation_set:(i+1)*size_of_validation_set]
                validation_labels = data_set.labels[i*size_of_validation_set:(i+1)*size_of_validation_set]

                training_data_set = DataSet(training_data, training_labels)
                validation_data_set = DataSet(validation_data, validation_labels)

                knn_model = KNNModel(training_data_set, self.K, self.distance_type, self.features_list)

                predictions = knn_model.evaluate_data_set(validation_data_set.data, normalize=normalize)

                evaluations.append(ModelAssessment(predictions, validation_data_set.labels).evaluation)

            average_evaluation = {k: sum(t[k] for t in evaluations)/len(evaluations) for k in evaluations[0]}
            model_evaluation = ModelAssessment()
            model_evaluation.set_evaluation(average_evaluation)
            self.evaluation = model_evaluation

        else:
            training_data_set = DataSet(data_set.data[size_of_validation_set:],
                                        data_set.labels[size_of_validation_set:])
            validation_data_set = DataSet(data_set.data[:size_of_validation_set],
                                          data_set.labels[:size_of_validation_set])

            knn_model = KNNModel(training_data_set, self.K, self.distance_type, self.features_list)

            predictions = knn_model.evaluate_data_set(validation_data_set.data, normalize=normalize)

            self.evaluation = ModelAssessment(predictions, validation_data_set.labels)

        if print_evaluation:
            self.evaluation.print_evaluation()

        return self.evaluation

    def clean_data_set(self, print_evolution=False, normalize=True):
        """
        This method "cleans" the training data set, it means that it tries to find a smaller data set that behave as
        closely as possible as the  training data set.
        The principle in to find the samples that are not the nearest neighbor to any other sample and delete them.
        :param print_evolution: (bool) if True then the evolution will be printed in the console
        :param normalize: (bool) if True then both the data set to predict and the training data set will be normalized
        :return: cleaned_data_set: (DataSet) new and cleaned training data set
        """

        if not isinstance(print_evolution, bool):
            raise TypeError("The argument 'print_evolution' must be a boolean")
        if not isinstance(normalize, bool):
            raise TypeError("The argument 'normalize' must be a boolean")

        def prediction(data, labels, test, k=1, distance="jaccard", features_list=None, normalize_=True):
            """
            Predicts the labels on the test set using the sets "data" and "labels" as training data set.
            :param data: (float array) training set of samples
            :param labels: (int array) training set of labels
            :param test: (float array) set of samples to classify
            :param k: (int) number of neighbors to use in the model
            :param distance: (string) "distance" type to use in the model
            :param features_list: (string list) list of the name of the features to use in the model
            :param normalize_: (bool) if True then both the data set to predict and the training data set are normalized
            :return: predictions, nearest_indices: results from the  evaluation on the test set using the
            "evaluate_data_set" method
            """
            data_set = DataSet(data, labels)
            try:
                knn_model = KNNModel(data_set, K=k, distance_type=distance, features_list=features_list)
                return knn_model.evaluate_data_set(test, normalize=normalize_, return_indices=True)
            except ValueError:
                if k >= data_set.N:
                    raise ValueError("The data set has become too small to predict its labels")
                else:
                    raise

        def get_loop(index_, nearest_indices_):
            """
            Gets the loop that starts at a given index in the graph of the nearest_neighbors
            :param index_: (int) index of the sample to start the loop (does not have any child in the graph)
            :param nearest_indices_: (int array) array of the nearest neighbor's index for each sample in the data set
            :return: loop_: (int list) list of the indices of the samples in the loop
            """
            loop_ = [index_]
            while nearest_indices_[loop_[-1]] not in loop_:
                loop_.append(nearest_indices_[loop_[-1]])
            return loop_

        original_data, original_labels = self.training_data_set.data, self.training_data_set.labels
        if self.predictions is None or self.nearest_indices is None:
            self.self_evaluation(normalize=normalize)
        cannot_delete = np.array([])
        nearest_indices = self.nearest_indices[:, 0]
        not_to_delete = [i for i in range(self.training_data_set.N)]

        relationship_dict = {}
        no_child = []
        for index in range(self.training_data_set.N):
            relationship_dict[index] = [x[0] for x in np.argwhere(nearest_indices == index)]
            if len(relationship_dict[index]) == 0:
                no_child.append(index)
        cannot_delete = np.unique(np.concatenate((cannot_delete, nearest_indices[no_child])))

        # Remove elements that have no child (i.e. that are not the closest neighbor to anyone)
        for index in np.intersect1d(no_child, not_to_delete):
            if index not in cannot_delete:
                loop = get_loop(index, nearest_indices)
                not_to_delete.remove(int(index))
                for x in loop[1:]:
                    cannot_delete = np.concatenate((cannot_delete, np.array([x])))

        while len(not_to_delete) <= self.K:
            for index in range(self.training_data_set.N):
                if index not in not_to_delete:
                    not_to_delete.append(index)

        cleaned_data, cleaned_labels = original_data[not_to_delete], original_labels[not_to_delete]
        cleaned_data_set = DataSet(cleaned_data, cleaned_labels)

        if print_evolution:
            new_predictions, new_nearest_indices = prediction(cleaned_data, cleaned_labels, original_data, k=self.K,
                                                              distance=self.distance_type,
                                                              features_list=self.features_list, normalize_=normalize)
            print("Cleaning done,\nSize of the new data set:", len(not_to_delete))
            if np.array_equal(self.predictions, new_predictions):
                print("No error")
            else:
                print("Errors:", len(np.argwhere(self.predictions[:, 0] != new_predictions[:, 0])[:, 0]))

            if self.training_data_set.C == 2:
                evaluation = ModelAssessment(new_predictions, self.training_data_set.labels)
                evaluation.print_evaluation()

        return cleaned_data_set

    def assess_cleaned_data_set(self, cleaned_data_set, normalize=True, print_evaluation=True):
        """
        Assess the predictions of a cleaned version of the training data set on the training data.
        :param cleaned_data_set: (DataSet) cleaned version of the training data set (or a different data set)
        :param normalize: (bool) if True then both the data set to predict and the training data set will be normalized
        :param print_evaluation: (bool) if True then prints the evaluation in the console
        :return: self.evaluation: (ModelAssessment) assessment of the cleaned data set
        """
        if not isinstance(cleaned_data_set, DataSet):
            raise TypeError("The argument 'cleaned_data_set' must be a DataSet")
        if not isinstance(normalize, bool):
            raise TypeError("The argument 'normalize' must be a boolean")
        if not isinstance(print_evaluation, bool):
            raise TypeError("The argument 'print_evaluation' must be a boolean")
        if cleaned_data_set.C != self.training_data_set.C:
            raise ValueError("The numbers of classes in the cleaned data set and the original data training data set "
                             "must match")
        if cleaned_data_set.C != 2:
            raise ValueError("The cleaned data set must have exactly two classes to be assessed")
        if cleaned_data_set.F != self.training_data_set.F and cleaned_data_set.F != self.F:
            raise ValueError("The number of features in the data set must match the number of features in the training "
                             "data set")

        knn_model = KNNModel(cleaned_data_set, K=self.K, distance_type=self.distance_type,
                             features_list=self.features_list)
        predictions, nearest_indices = knn_model.evaluate_data_set(self.training_data_set.data,
                                                                   normalize=normalize, return_indices=True)
        self.evaluation = ModelAssessment(predictions, self.training_data_set.labels)
        if print_evaluation:
            # self.evaluation.print_evaluation()
            pass
        return self.evaluation

    def add_sample_to_database(self, features, label, predictions=None, normalize=True):
        """
        Gives a criterion to choose if a recording should be added to the training data set or not.
        If the training data set predicts correctly the labels on each segment (i.e. feature set in "features")
        then the recording's features will not be added to the data set
        :param features: (float array) set of features on each segment of the recording to add
        :param label: (int) label of the recording to add
        :param predictions: (int array) predictions of the training data set on the sample
        :param normalize: (bool) if True then both the data set to predict and the training data set will be normalized
        :return: addition_criterion: (bool) criterion to add the features to the data set or not.
        """

        if not isinstance(normalize, bool):
            raise TypeError("The argument 'normalize' must be a boolean")
        if not isinstance(features, np.ndarray):
            raise TypeError("The features matrix must be a numpy array")
        if len(np.shape(features)) != 2:
            raise ValueError("The features matrix must have exactly two dimensions")
        if len(features) == 0:
            raise ValueError("The features matrix provided is empty")
        if len(features[0]) < 1:
            raise ValueError("The number of features in features matrix must be at least one")
        if len(features[0]) != self.training_data_set.F and len(features[0]) != self.F:
            raise ValueError("The number of features in the features matrix and in the training data set must match")
        if not isinstance(label, int) or label < 0:
            raise TypeError("The label must be a positive integer")
        if label >= self.training_data_set.C:
            raise ValueError("The label must not exceed or be equal to the number of classes in the training data set")

        formalized_label = np.zeros((1, self.training_data_set.C))
        formalized_label[0][int(label)] = 1
        n = len(features)
        labels = np.repeat(formalized_label, n, axis=0).astype(np.int)
        # If the predictions argument has not been provided or is incorrect then compute it using the training data set
        if predictions is None or not isinstance(predictions, np.ndarray) or np.shape(predictions) != np.shape(labels):
            predictions = self.evaluate_data_set(features, normalize=normalize)
        addition_criterion = not np.array_equal(predictions, labels)

        return addition_criterion

import numpy as np


class ModelAssessment:

    def __init__(self, predictions=None, labels=None):

        if (predictions is not None) and (labels is not None):
            if not isinstance(predictions, np.ndarray):
                raise TypeError("The predictions must be a numpy array")
            if not isinstance(labels, np.ndarray):
                raise TypeError("The set of labels must be a numpy array")
            if len(predictions) == 0:
                raise ValueError("The prediction array must ot be empty")
            if len(labels) == 0:
                raise ValueError("The set of labels must ot be empty")
            if len(np.shape(predictions)) != 2:
                raise ValueError("The array of predictions must have exactly two dimensions")
            if len(np.shape(labels)) != 2:
                raise ValueError("The set of labels must have exactly two dimensions")
            if len(predictions[0]) != 2:
                raise ValueError("The prediction array must have exactly two classes")
            if len(labels[0]) != 2:
                raise ValueError("The set of labels must have exactly two classes")
            if np.shape(predictions) != np.shape(labels):
                raise ValueError("Both predictions and labels must have the same dimensions")
        if (predictions is None and labels is not None) or (predictions is not None and labels is None):
            raise ValueError("One of the arguments cannot be 'None' if the other one is")

        self.predictions = predictions
        self.labels = labels
        self.evaluation = {}
        self.evaluate()

    def set_evaluation(self, evaluation):
        """
        Sets the evaluation to a predefined dictionary
        :param evaluation: (dictionary) predefined dictionary
        :return: None
        """
        self.evaluation = evaluation

    def evaluate(self):
        """
        Evaluates the model according to the criteria of the 2016 PhysioNet Challenge:
        https://physionet.org/content/challenge-2016/1.0.0/
        :return: None
        """

        if self.predictions is not None and self.labels is not None:
            (N, C) = np.shape(self.labels)

            negative = np.argwhere(self.predictions[:, 0] == np.ones(N))
            if len(negative) > 0:
                true_negative = np.argwhere(self.predictions[negative, 0] == self.labels[negative, 0])
                tn = len(true_negative)/len(negative)
                fn = (len(negative) - len(true_negative))/len(negative)
            else:
                tn = 0
                fn = 0
            self.evaluation["TN"] = tn
            self.evaluation["FN"] = fn

            positive = np.argwhere(self.predictions[:, 1] == np.ones(N))
            if len(positive) > 0:
                true_positive = np.argwhere(self.predictions[positive, 1] == self.labels[positive, 1])
                tp = len(true_positive)/len(positive)
                fp = (len(positive) - len(true_positive))/len(positive)
            else:
                tp = 0
                fp = 0
            self.evaluation["TP"] = tp
            self.evaluation["FP"] = fp

            n = tn + fp
            p = tp + fn

            recall = tp/max(p, 1e-12)
            specificity = tn/max(n, 1e-12)
            fpr = 1 - specificity   # False Positive Rate
            precision = tp/max((tp + fp), 1e-12)
            f_score = 2*recall*precision/max((recall + precision), 1e-12)
            accuracy = (tp + tn)/max((n + p), 1e-12)
            accuracy_normal = tn/max(n, 1e-12)
            accuracy_abnormal = tp/max(p, 1e-12)
            error = 1 - accuracy
            mcc = (tp*tn - fp*fn)/max((((tp + fp)*n*p*(tn + fn))**0.5), 1e-12)  # Matthews correlation coefficient

            self.evaluation["Accuracy"] = accuracy
            self.evaluation["Accuracy normal"] = accuracy_normal
            self.evaluation["Accuracy abnormal"] = accuracy_abnormal
            self.evaluation["Error"] = error
            self.evaluation["Sensitivity"] = recall
            self.evaluation["Specificity"] = specificity
            self.evaluation["Precision"] = precision
            self.evaluation["FPR"] = fpr
            self.evaluation["F_Score"] = f_score
            self.evaluation["MCC"] = mcc

    def print_evaluation(self):
        """
        Prints the evaluation
        :return: None
        """

        if len(self.evaluation) == 0:
            self.evaluate()

        print("True negative", self.evaluation["TN"], "False negative", self.evaluation["FN"])
        print("True positive", self.evaluation["TP"], "False positive", self.evaluation["FP"])

        print("Accuracy:", self.evaluation["Accuracy"])
        print("Accuracy normal:", self.evaluation["Accuracy normal"])
        print("Accuracy abnormal:", self.evaluation["Accuracy abnormal"])
        print("Error:", self.evaluation["Error"])
        print("Sensitivity:", self.evaluation["Sensitivity"])
        print("Specificity:", self.evaluation["Specificity"])
        print("Precision:", self.evaluation["Precision"])
        print("FPR:", self.evaluation["FPR"])
        print("F_Score:", self.evaluation["F_Score"])
        print("MCC:", self.evaluation["MCC"])

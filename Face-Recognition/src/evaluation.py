import pickle

import numpy as np

from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        similarity_thresholds = []
        identification_rates = []

        self.classifier.fit(self.train_embeddings, self.train_labels)

        pred_labels, similarities = self.classifier.predict_labels_and_similarities(
            self.test_embeddings
        )

        similarities = np.asarray(similarities)
        pred_labels = np.asarray(pred_labels)

        for far in self.false_alarm_rate_range:
            thr = self.select_similarity_threshold(similarities, far)
            similarity_thresholds.append(thr)

            open_set_predictions = pred_labels.copy()
            open_set_predictions[similarities < thr] = UNKNOWN_LABEL

            ir = self.calc_identification_rate(open_set_predictions)
            identification_rates.append(ir)

        # Report all performance measures.
        evaluation_results = {
            "similarity_thresholds": np.asarray(similarity_thresholds),
            "identification_rates": np.asarray(identification_rates),
            "false_alarm_rates": self.false_alarm_rate_range,
        }

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        similarity = np.asarray(similarity)
        target_labels = np.asarray(self.test_labels)

        unknown_mask = target_labels == UNKNOWN_LABEL
        unknown_similarities = similarity[unknown_mask]

        if len(unknown_similarities) == 0:
            return np.inf

        percentile = 100.0 * (1.0 - false_alarm_rate)
        return np.percentile(unknown_similarities, percentile)

    def calc_identification_rate(self, prediction_labels):
        prediction_labels = np.asarray(prediction_labels)
        target_labels = np.asarray(self.test_labels)

        known_mask = target_labels != UNKNOWN_LABEL
        if np.sum(known_mask) == 0:
            return 0.0

        correct_pred = prediction_labels[known_mask] == target_labels[known_mask]
        return np.mean(correct_pred)

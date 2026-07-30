import time
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, normalize
from sklearn.cluster import KMeans

from config import Config

import warnings
warnings.filterwarnings("ignore")


UNKNOWN_LABEL: Final[int] = -1


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    x_train = x_train.astype(np.float64)

    # scaling the input data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)


    # Training the preprocessed data using a classifier algorithm
    model = LogisticRegression(C=10.0, class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)

    model.fit(x_train, y_train)


    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        # apply preprocessing steps
        x_test = x_test.astype(np.float64)
        x_test = scaler.transform(x_test)

        # predict the labels using our trained model
        y_pred = model.predict(x_test)

        # get the confidence score for each class
        probs = model.predict_proba(x_test)

        # calculating y scores
        known_mask = model.classes_ != UNKNOWN_LABEL
        known_probs = probs[:, known_mask]
        y_score = np.max(known_probs, axis=1)

        # using a threshold for our prediction based on the highest confidence score
        y_pred[y_score < 0.5] = UNKNOWN_LABEL

        return y_pred, y_score

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # Preprocessing steps:
    x_train = x_train.astype(np.float64)

    #Scaling data using standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)


    # split the data into known and unknown samples using mask for each group
    known_mask = y_train != UNKNOWN_LABEL
    unknown_mask = ~known_mask

    x_train_known = x_train[known_mask]
    x_train_unknown = x_train[unknown_mask]
    y_train_known = y_train[known_mask]
    y_train_unknown = y_train[unknown_mask]

    max_label = np.max(y_train)

    # assigning new labels to our unknown samples instead of just one class (-1)
    # this can be done by using a clustering algorithm
    x_train_unknown_norm = normalize(x_train_unknown, norm='l2')
    clustering_model = KMeans(15, random_state=42, n_init=20)
    y_clustered = clustering_model.fit_predict(x_train_unknown_norm)

    # reassign unkonwn labels and add the maximum label to cluster number to avoid overlapping with known classes
    y_train_unknown = max_label + y_clustered + 1

    # concatenate known and unknown data together into a one signle dataset
    x_train = np.concatenate((x_train_known, x_train_unknown))
    y_train = np.concatenate((y_train_known, y_train_unknown))


    # Train a classification model on our preprocessed data
    model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)


    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        # apply preprocessing steps
        x_test = x_test.astype(np.float64)
        x_test = scaler.transform(x_test)

        # predict the labels using our trained model
        y_pred = model.predict(x_test)
        probs = model.predict_proba(x_test)
        y_pred[y_pred > max_label] = UNKNOWN_LABEL

        # calculating y scores
        known_mask = model.classes_ <= max_label
        known_probs = probs[:, known_mask]
        y_score = np.max(known_probs, axis=1)

        y_pred[y_score < 0.1] = UNKNOWN_LABEL

        return y_pred, y_score

    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    # Load our data
    X, y = load_challenge_train_data()

    # using cross validation to test our data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # train our data using both algorithms that we implemented: spl and mpl
    for training_fn in (spl_training, mpl_training):

        # define arrays to keep our metrics
        fit_times, predict_times = [], []
        auc_scores, balanced_ranks = [], []
        dirs_1, dirs_10 = [], []
        unknown_accs = []

        # loop over each fold separately and train the data
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):

            # get x and y splits of each fold
            x_train, y_train = X[train_index], y[train_index]
            x_test, y_test = X[test_index], y[test_index]

            # Fit our spl or mpl algorithms and measure the training
            t0_fit = time.time()
            predict_fn = training_fn(x_train, y_train)
            t1_fit = time.time()

            fit_time = t1_fit - t0_fit
            fit_times.append(fit_time)

            # Predict the labels using trained model and calculate the prediction time per sample
            t0_pred = time.time()
            y_pred, y_score = predict_fn(x_test)
            t1_pred = time.time()

            predict_time = t1_pred - t0_pred
            sample_pred_time = predict_time / len(x_test) * 1000   # in millisecond
            predict_times.append(sample_pred_time)

            # estimate area under the curve as one of the metrics
            # as roc_auc_score just accept binary labels so first we need to convert our labels to known or unknown labels
            y_test_binary = y_test.copy()
            y_test_binary[y_test_binary != UNKNOWN_LABEL] = 1
            y_test_binary[y_test_binary == UNKNOWN_LABEL] = 0
            aucroc = roc_auc_score(y_test_binary, y_score)
            auc_scores.append(aucroc)


            # Calculate balanced accuracy on known data
            known_mask = y_test != UNKNOWN_LABEL
            balanced_acc = balanced_accuracy_score(y_test[known_mask], y_pred[known_mask])
            balanced_ranks.append(balanced_acc)


            unknown_mask = y_test == UNKNOWN_LABEL
            unknown_acc = (y_pred[unknown_mask] == UNKNOWN_LABEL).mean()
            unknown_accs.append(unknown_acc)


            # Calculate DIR@FAR for both 1% and 10%
            quant_1 = np.quantile(y_score[unknown_mask], 0.99) # for 1%
            quant_10 = np.quantile(y_score[unknown_mask], 0.9) # for 10%

            tp_mask = y_pred[known_mask] == y_test[known_mask]  # true positive mask
            mask_quant_1 = y_score[known_mask] >= quant_1
            mask_quant_10 = y_score[known_mask] >= quant_10

            dir_1 = np.sum(tp_mask & mask_quant_1)/ len(y_test[known_mask])
            dir_10 = np.sum(tp_mask & mask_quant_10) / len(y_test[known_mask])

            dirs_1.append(dir_1)
            dirs_10.append(dir_10)


        print("#" * 30)
        print(training_fn.__name__)
        print("#" * 30)
        print(f"Fitting time (s): {np.mean(fit_times):.4f} seconds")
        print(f"Prediction time per sample (ms): {np.mean(predict_times):.4f} milliseconds")
        print(f"AUCROC: {np.mean(auc_scores):.4f} with {np.std(auc_scores)}")
        print(f"Balanced accuracy: {np.mean(balanced_ranks):.4f}")
        print(f"DIR@FAR=1%: {np.mean(dirs_1):.4f} ")
        print(f"DIR@FAR=10%: {np.mean(dirs_10):.4f}")
        print(f"Unknown rejection rate: {np.mean(unknown_accs):.4f}")
        print("\n")

if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import TaggedDocument
from hdbscan import HDBSCAN, all_points_membership_vectors


def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = '%03d-results.tsv':

    001-results.tsv
    001-results.tsv
    001-results.tsv

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while Path(path_pattern % i).exists():
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if Path(path_pattern % c).exists() else (a, c)

    return path_pattern % b


def save_data(results_df, data_params, param_str, sort_by, res_folder="/home/philipp/projects/dad4td/reports/density_estimation/", res_pattern="%04d_dens_eval.tsv"):
    result_folder = res_folder
    results_path = next_path(result_folder + res_pattern)
    results_param_path = result_folder + Path(results_path).stem + ".txt"

    save_results(results_df, results_path, data_params, sort_by)
    save_params(results_param_path, param_str)


def save_results(results_df, results_path, data_params, sort_by):
    out_df = pd.DataFrame.from_dict(results_df)
    for key, val in data_params.items():
        out_df[key] = val
    out_df = out_df.sort_values(by=[sort_by])
    out_df.to_csv(results_path, sep="\t")


def save_params(params_path, param_str):
    with open(params_path, 'w') as res_params:
        res_params.write(param_str)


class TaggedDocsTransformer(BaseEstimator, TransformerMixin):
    """
    a general class for creating a machine learning step in the machine learning pipeline
    """

    def __init__(self, lower=False):
        """
        constructor
        """
        self.lower = lower

    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        return self

    def transform(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        if self.lower:
            X = X.str.lower()
        return [TaggedDocument(doc, str(i)) for i, doc in X.items()]


class GLOSHTransformer(BaseEstimator, TransformerMixin):
    """
    a general class for creating a machine learning step in the machine learning pipeline
    """

    def __init__(self):
        """
        constructor
        """
        pass

    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        return self

    def predict(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        clusterer = HDBSCAN(min_cluster_size=2).fit(X)
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
        outliers = np.where(clusterer.outlier_scores_ > threshold, -1, 1)
        out_print = clusterer.labels_
        print(threshold)
        print(out_print)
        print(len(out_print))
        print(out_print.shape)
        unique_elements, counts_elements = np.unique(
            out_print, return_counts=True)
        print("Frequency of unique values of the said array:")
        print(np.asarray((unique_elements, counts_elements)))
        return outliers


class HDBSCANPredictor(BaseEstimator, TransformerMixin):

    def __init__(self, min_cluster_size=5, metric="euclidean", no_noise=False):
        self.no_noise = no_noise
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X, y=None, **kwargs):
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            metric=self.metric, prediction_data=self.no_noise)
        if self.no_noise:
            try:
                clusterer = clusterer.fit(X)
                X = np.argmax(
                    all_points_membership_vectors(clusterer)[:, 1:], axis=1)
            except IndexError:
                print("IndexError for this run. Using clustering with noise.")
                X = clusterer.fit_predict(X)
        else:
            X = clusterer.fit_predict(X)
        return X

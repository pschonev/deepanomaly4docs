from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import TaggedDocument


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


def sample_data(df, data_frac, contamination, seed):
    X_n = int(df.shape[0] * data_frac)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    df = df.reset_index(drop=True)
    return df


def save_data(results_df, data_params, param_str):
    result_folder = "/home/philipp/projects/dad4td/reports/"
    results_path = next_path(result_folder + "%04d_dens_eval.tsv")
    results_param_path = result_folder + Path(results_path).stem + ".txt"

    save_results(results_df, results_path, data_params)
    save_params(results_param_path, param_str)


def save_results(results_df, results_path, data_params):
    out_df = pd.DataFrame.from_dict(results_df)
    for key, val in data_params.items():
        out_df[key] = val
    out_df = out_df.sort_values(by=['rank_test_f1_macro'])
    out_df.to_csv(results_path, sep="\t")


def save_params(params_path, param_str):
    with open(params_path, 'w') as res_params:
        res_params.write(param_str)


class TaggedDocsTransformer(BaseEstimator, TransformerMixin):
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

    def transform(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        return [TaggedDocument(doc, str(i)) for i, doc in X.items()]

import pandas as pd
import numpy as np


def sample_data(df, data_frac, contamination, seed=42):
    X_n = int(df.shape[0] * data_frac)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    df = df.reset_index(drop=True)
    return df


def get_out_data(dataset_name, data_frac, contamination, seed=42):
    datasets = {
        "imdb_20news": "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"}

    # prepare data
    # ! creation of datasets should probably be handled seperately all within a class/function
    df = pd.read_pickle(datasets[dataset_name])
    # class for imdb_20news that lets me choose 20 news categories?
    df = sample_data(df, data_frac, contamination)

    data_text = df["text"]
    outlier_labels = df["outlier_label"]

    return data_text, outlier_labels

from pathlib import Path
from itertools import product
from sklearn.metrics import roc_auc_score, homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score, accuracy_score
import numpy as np

def remove_short_texts(df, data_name, min_len):
    bef = df.shape[0]
    df["text_len"] = df.text.map(lambda x: len(x))
    df = df.where(df.text_len > min_len).dropna().reset_index(drop=True)
    print(
        f"Removed {bef-df.shape[0]} rows from {data_name} because they were under {min_len} characters long.")
    return df


def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = '%03d-results.tsv':

    001-results.tsv
    001-results.tsv
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


def product_dict(**kwargs):
    return [dict(zip(kwargs.keys(), x)) for x in product(*kwargs.values())]


def get_scores(scores, outlier_labels, decision_scores, inlabel=1, outlabel=-1, threshold=0.5):

    print(decision_scores)
    outlier_pred = np.where(decision_scores > threshold, 1, 0)
    scores[f"f1"] = f1_score(
        outlier_labels, outlier_pred)
    scores[f"f1_macro"] = f1_score(
        outlier_labels, outlier_pred, average='macro')
    scores[f"in_f1"] = f1_score(
        outlier_labels, outlier_pred, pos_label=inlabel)
    scores[f"in_rec"] = recall_score(
        outlier_labels, outlier_pred, pos_label=inlabel)
    scores[f"in_prec"] = precision_score(
        outlier_labels, outlier_pred, pos_label=inlabel)
    scores[f"out_f1"] = f1_score(
        outlier_labels, outlier_pred, pos_label=outlabel)
    scores[f"out_rec"] = recall_score(
        outlier_labels, outlier_pred, pos_label=outlabel)
    scores[f"out_prec"] = precision_score(
        outlier_labels, outlier_pred, pos_label=outlabel)
    scores[f"accuracy"] = accuracy_score(
        outlier_labels, outlier_pred)
    scores[f"auroc"] = roc_auc_score(
        decision_scores, outlier_pred)
    return scores


def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = np.quantile(sr, [pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return ((np.abs(sr - median)) >= iqr/2), median, iqr


def sample_data(df, fraction, contamination, seed):
    X_n = int(df[df.outlier_label == 1].shape[0] * fraction)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    df = df.reset_index(drop=True)
    return df

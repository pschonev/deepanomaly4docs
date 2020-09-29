# %%
import numpy as np
import pandas as pd
from umap import UMAP
from ivis import Ivis
from numpy import percentile
from collections import defaultdict
from eval_cluster_config import Doc2VecModel
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score


def sample_data(df, fraction, contamination, seed):
        X_n = int(df.shape[0] * fraction)
        y_n = int(X_n * contamination)

        df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
        df = df[df["outlier_label"] == 1].head(X_n).append(
            df[df["outlier_label"] == -1].head(y_n))
        df = df.reset_index(drop=True)
        return df

def get_scores(scores, outlier_labels, outlier_pred):
    scores[f"f1_macro"] = f1_score(
        outlier_labels, outlier_pred, average='macro')
    scores[f"in_f1"] = f1_score(
        outlier_labels, outlier_pred, pos_label=1)
    scores[f"out_f1"] = f1_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_rec"] = recall_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_prec"] = precision_score(
        outlier_labels, outlier_pred, pos_label=-1)
    return scores

seed = 42
fraction = 1.0
contamination = 0.1

data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
scores = defaultdict(list)

df = pd.read_pickle(data_path)
df = sample_data(df, fraction, contamination, seed)

df.columns

#%%
doc2vec_path = "/home/philipp/projects/dad4td/models/all_news_05_30_30/all_news.bin"
doc2vec_model = Doc2VecModel("all_news_05_30_30", "all_news", 0.5,
                                           30, 30, doc2vec_path)
docvecs = doc2vec_model.vectorize(df["text"])
#%%
# UMAP
dim_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                   n_components=1, random_state=42)

docvecs = list(docvecs)
dim_reduced_vecs = dim_reducer.fit_transform(docvecs)

decision_scores = dim_reduced_vecs.astype(float)

#%%
# Ivis
dim_reducer = Ivis(embedding_dims=1, k=15, model="maaten", n_epochs_without_progress=15)
docvecs = np.vstack(df["doc2vecwikiimdb20news013030_256"].to_numpy())
dim_reduced_vecs = dim_reducer.fit_transform(docvecs)
decision_scores = dim_reduced_vecs.astype(float)

#%%
# Read saved from DF
df = pd.read_pickle(data_path)
df = sample_data(df, fraction, contamination, 44)
decision_scores = df["apnews_1"].astype(float).to_numpy()

#%%
# Get outlier score
def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = np.quantile(sr, [pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    print(qlow, median, qhigh)
    return ((np.abs(sr - median)) >= iqr/2)

preds = reject_outliers(decision_scores, iq_range=1.0-contamination)
preds = [-1 if x else 1 for x in preds]

scores = get_scores(scores, df["outlier_label"], preds)
scores

#%%
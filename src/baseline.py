# %%
import numpy as np
import pandas as pd
from umap import UMAP
from ivis import Ivis
from numpy import percentile
from collections import defaultdict
from evaluation import Doc2VecModel
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from evaluation import get_scores, reject_outliers, sample_data

tqdm.pandas(desc="progess: ")

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
                   n_components=256, random_state=42)

dim_reduced_vecs = dim_reducer.fit_transform(list(docvecs))
decision_scores = dim_reduced_vecs.astype(float)

#%%
# Ivis
dim_reducer = Ivis(embedding_dims=1, k=15, model="maaten", n_epochs_without_progress=15)
docvecs = np.vstack(df["apnews_256"].to_numpy())
dim_reduced_vecs = dim_reducer.fit_transform(docvecs)
decision_scores = dim_reduced_vecs.astype(float)

#%%
# Read saved from DF
df = pd.read_pickle(data_path)
df = sample_data(df, fraction, contamination, 44)
decision_scores = df["apnews_1"].astype(float).to_numpy()

#%%
# Get outlier score

preds = reject_outliers(decision_scores, iq_range=1.0-contamination)
preds = [-1 if x else 1 for x in preds]

scores = get_scores(scores, df["outlier_label"], preds)
scores

#%%
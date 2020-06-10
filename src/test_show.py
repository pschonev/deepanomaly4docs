# %%
import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from visualization.visualize import create_show_graph
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer


def sample_data(df, data_frac, contamination, seed):
    X_n = int(df.shape[0] * data_frac)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    print(df)
    return df


# parameters
data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"

d = dict(data_frac=0.1,
         contamination=0.1,
         seed=42)

# prepare data
df = pd.read_pickle(data_path)
df = sample_data(df, **d)


X = df["text"]
y = df["outlier_label"]

# pipeline
print("TF-IDF ...")
tfidf_vecs = TfidfVectorizer(min_df=25, stop_words='english').fit_transform(X)
print("dim reduction ...")
dim_reduced_vecs = UMAP(metric="euclidean", set_op_mix_ratio=0.0,
                  n_components=50, random_state=42).fit_transform(tfidf_vecs)
print("dim reduction 2D ...")
vecs_2d = UMAP(metric="euclidean", set_op_mix_ratio=0.0,
                  n_components=2, random_state=42).fit_transform(tfidf_vecs)
print("Local outlier factor ...")
df["predicted"] = LocalOutlierFactor(
    novelty=False, metric="euclidean", contamination=d["contamination"]).fit_predict(dim_reduced_vecs)

create_show_graph(df, "text", coords_2d=vecs_2d, color="predicted")
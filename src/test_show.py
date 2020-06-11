# %%
import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from visualization.visualize import create_show_graph
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from gensim.sklearn_api import D2VTransformer
from gensim.models.doc2vec import TaggedDocument

def sample_data(df, data_frac, contamination, seed):
    X_n = int(df.shape[0] * data_frac)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    df = df.reset_index(drop=True)
    return df


def get_result(row):
    if row["outlier_label"] == 1 and row["predicted"] == 1:
        return "inlier - true positive"
    if row["outlier_label"] == -1 and row["predicted"] == -1:
        return "outlier - true negative"
    if row["outlier_label"] == -1 and row["predicted"] == 1:
        return "false negative (outlier predicted as inlier)"
    if row["outlier_label"] == 1 and row["predicted"] == -1:
        return "false positive (inlier predicted as outlier)"
    return "-1"


# parameters
data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"

d = dict(data_frac=0.1,
         contamination=0.1,
         seed=42)

# prepare data
df = pd.read_pickle(data_path)

# doc2vec model
print("create doc2vec model..")
d2v_model = doc_vecs = D2VTransformer(seed=d["seed"], min_count=25, size=300, window=5).fit(df["text"])

# sample
df = sample_data(df, **d)


X = df["text"]
X_tagged = [TaggedDocument(doc, str(i)) for i, doc in df["text"].items()]

# pipeline
print("Doc2Vec ...")
doc_vecs = d2v_model.transform(X_tagged)
#tfidf_vecs = TfidfVectorizer(min_df=25, stop_words='english').fit_transform(X)
print("dim reduction ...")
dim_reduced_vecs = UMAP(metric="euclidean", set_op_mix_ratio=0.0,
                        n_components=100, random_state=42).fit_transform(doc_vecs)
print("dim reduction 2D ...")
vecs_2d = UMAP(metric="euclidean", set_op_mix_ratio=1.0,
               n_components=2, random_state=42).fit_transform(doc_vecs)
print("Local outlier factor ...")
df["predicted"] = LocalOutlierFactor(
    novelty=False, metric="euclidean", contamination=d["contamination"]).fit_predict(dim_reduced_vecs)

df["result"] = df.apply(lambda row: get_result(row), axis=1)


title = df["result"].value_counts().to_string().replace("\n", "\t")
print(classification_report(df["outlier_label"], df["predicted"]))

fig = create_show_graph(df, "text", coords_2d=vecs_2d, color="result")
fig.update_layout(title=title)
fig.show()

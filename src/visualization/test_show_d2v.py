# %%
import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from hdbscan import HDBSCAN, all_points_membership_vectors
from visualize import create_show_graph
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, homogeneity_score
from gensim.sklearn_api import D2VTransformer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize
from tqdm import tqdm

tqdm.pandas(desc="my bar!")


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
model_path = "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin"

d = dict(data_frac=0.4,
         contamination=0.1,
         seed=42)

showclusters = True
allow_noise = False
min_doc_length = 5
n_comps = 10
min_cluster_size = 8

# prepare data
print("Get data...")
df = pd.read_pickle(data_path)
n_before = df.shape[0]
df = df[df['text'].map(len) > min_doc_length]
print(f"Removed {n_before - df.shape[0]} rows with doc length below {min_doc_length}.")
#all_docs = df["text"]

# sample
df = sample_data(df, **d)

X = df["text"]

# text lowered and split into list of tokens
print("Pre-process data...")
#all_docs = all_docs.progress_apply(lambda x: simple_preprocess(x))
X = X.progress_apply(lambda x: simple_preprocess(x))

print("TaggedDocuments being prepared...")
#all_docs_tagged = [TaggedDocument(doc, [i]) for i, doc in all_docs.items()]
tagged_docs = [TaggedDocument(doc, [i]) for i, doc in X.items()]

print("Train Doc2Vec model...")
#model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
#model.build_vocab(all_docs_tagged)
model = Doc2Vec.load(model_path)

print("Infer doc vectors...")
docvecs = X.progress_apply(lambda x: model.infer_vector(x))
docvecs = list(docvecs)

#docvecs = docvecs.to_numpy()
#print("dim reduction ...")
dim_reduced_vecs = UMAP(metric="cosine", set_op_mix_ratio=0,
                        n_components=n_comps, random_state=42).fit_transform(docvecs)

print("dim reduction 2D ...")
vecs_2d = UMAP(metric="cosine", set_op_mix_ratio=0,
               n_components=2, random_state=42).fit_transform(docvecs)

#print("Local outlier factor ...")
#df["predicted"] = LocalOutlierFactor(
#    novelty=False, metric="euclidean", contamination=d["contamination"]).fit_predict(dim_reduced_vecs)

print("HDBSCAN ...")
#dim_reduced_vecs = normalize(dim_reduced_vecs, norm="l2")
clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, metric="euclidean").fit(dim_reduced_vecs)
threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
df["predicted"] = np.where(clusterer.outlier_scores_ > threshold, -1, 1)


df["result"] = df.apply(lambda row: get_result(row), axis=1)


title = df["result"].value_counts().to_string().replace("\n", "\t")
title = f"m_clus: {clusterer.min_cluster_size} n_comp: {n_comps}" + title
print(classification_report(df["outlier_label"], df["predicted"]))
outlier_labels = df["outlier_label"]
print(all_points_membership_vectors(clusterer))
cluster_labels = clusterer.labels_ if allow_noise else np.argmax(
    all_points_membership_vectors(clusterer)[:, 1:], axis=1)
print(f"\nHomogeneity: {homogeneity_score(outlier_labels, cluster_labels)}")
crosstab = pd.crosstab(cluster_labels, outlier_labels, normalize='index')
print(f"\n\n {crosstab}")
crosstab_abs = pd.crosstab(cluster_labels, outlier_labels)
print(f"\n\n {crosstab_abs}")

if showclusters:
    df["result"] = cluster_labels.astype(str)
fig = create_show_graph(df, "text", coords_2d=vecs_2d, color="result")
fig.update_layout(title=title)
fig.show()

# !! get imdb % of each cluster and homogeneity score

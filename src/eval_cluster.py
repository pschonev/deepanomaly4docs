# %%
import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score
from gensim.sklearn_api import D2VTransformer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize
from itertools import product
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from eval_utils import next_path

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
result_folder = "/home/philipp/projects/dad4td/reports/clustering/"
res_pattern = "%04d_cluster_eval_new.tsv"
result_path = next_path(result_folder + res_pattern)

data_params = dict(data_frac=0.3,
                   contamination=0.1,
                   seed=42)

allow_noise = False
min_doc_length = 5

iter_params = OrderedDict(n_comps_=[10, 20], mix_ratios_=[0.0, 0.5],
                          umap_metrics_=["cosine"], min_cluster_sizes_=[8])

# load data and remove empty texts
print("Get data...")
df = pd.read_pickle(data_path)
n_before = df.shape[0]
df = df[df['text'].map(len) > min_doc_length]
print(
    f"Removed {n_before - df.shape[0]} rows with doc length below {min_doc_length}.")

# sample
df = sample_data(df, **data_params)
X = df["text"]

# text lowered and split into list of tokens
print("Pre-process data...")
X = X.progress_apply(lambda x: simple_preprocess(x))

print("TaggedDocuments being prepared...")
tagged_docs = [TaggedDocument(doc, [i]) for i, doc in X.items()]


#model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(all_docs_tagged)
print("Load Doc2Vec model...")
model = Doc2Vec.load(model_path)

print("Infer doc vectors...")
docvecs = X.progress_apply(lambda x: model.infer_vector(x))
docvecs = list(docvecs)

scores = defaultdict(list)
result_df = pd.DataFrame()
for params in tqdm(product(*iter_params.values())):
    n_comp, mix_ratio, umap_metric, min_cluster_size = params
    print(f"n_comp {n_comp}, mix_ratio: {mix_ratio}, umap_metric: {umap_metric}, min_cluster_size: {min_cluster_size}")
    dim_reduced_vecs = UMAP(metric=umap_metric, set_op_mix_ratio=mix_ratio,
                            n_components=n_comp, random_state=42).fit_transform(docvecs)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                        prediction_data=True, metric="euclidean").fit(dim_reduced_vecs)

    # GLOSH
    threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
    df["predicted"] = np.where(clusterer.outlier_scores_ > threshold, -1, 1)

    # cluster scoring
    outlier_labels = df["outlier_label"]
    outlier_pred = df["predicted"]
    cluster_pred = clusterer.labels_ if allow_noise else np.argmax(
        all_points_membership_vectors(clusterer)[:, 1:], axis=1)

    # adding param values to results dict
    for key, value in zip(iter_params, params):
        scores[key] = value

    scores["homogeneity"] = homogeneity_score(outlier_labels, cluster_pred)
    scores["completeness"] = completeness_score(outlier_labels, cluster_pred)
    scores["v_measure"] = v_measure_score(outlier_labels, cluster_pred)
    scores["f1_macro"] = f1_score(
        outlier_labels, outlier_pred, average='macro')

    result_df = result_df.append(scores, ignore_index=True)
    results_df = result_df.sort_values(by=["homogeneity"]).reset_index(drop=True)
    results_df.to_csv(result_path, sep="\t")

print(result_df)
# crosstabs
#crosstab = pd.crosstab(cluster_labels, outlier_labels, normalize='index')
#print(f"\n\n {crosstab}")
#crosstab_abs = pd.crosstab(cluster_labels, outlier_labels)
#print(f"\n\n {crosstab_abs}")

# %%
from numpy import percentile
from timeit import default_timer as timer
from collections import defaultdict
from eval_utils import next_path
from tqdm import tqdm
import pandas as pd
import plotly.express as px
from umap import UMAP
from eval_cluster_config import TestData, Doc2VecModel, PyodDetector
from pyod.models.ocsvm import OCSVM

tqdm.pandas(desc="progess: ")

# parameters
set_op_mix_ratio = 1.0
n_components = 256
data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
test_data = TestData(
    data_path, "imdb_20news", fraction=[], contamination=[], seed=[])
doc2vec_model = Doc2VecModel("doc2vecwikiimdb20news013030", "wiki_EN_imdb_20news", 0.1,
                                           30, 30, "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_30_min30/doc2vec_wiki.bin")
dim_reducer = UMAP(metric="cosine", set_op_mix_ratio=set_op_mix_ratio,
                   n_components=n_components, random_state=42)


# %% save vectors
vector_name = "doc2vecwikiimdb20news013030"
save_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"

test_data.load_data()
df = test_data.df
docvecs = doc2vec_model.vectorize(df["text"])
df[vector_name] = docvecs
df.to_pickle(save_path)
# %%
print(test_data.df.shape[0])
test_data.remove_short_texts()
print(test_data.df.shape[0])
df
#df = test_data.sample_data(fraction=1.0, contamination=0.1, seed=1)

# %%
test_data.load_data()
df = test_data.df
vector_name = "doc2vecwikiimdb20news013030"
docvecs = df[vector_name]
docvecs
# %%
# dim reduce
dim_reduced_vecs = dim_reducer.fit_transform(list(docvecs))
df["doc2vecwikiimdb20news013030_256"] = list(dim_reduced_vecs)
df
# %%
df.to_pickle(data_path)
# %%
# outlier prediction
scores = defaultdict(list)
outlier_predictor = PyodDetector(OCSVM, "OCSVM")
scores, preds = outlier_predictor.predict(
    dim_reduced_vecs, scores, test_data.df["outlier_label"], 0.1, "OCSVM")

# %%
# baseline using only UMAP

df = pd.read_pickle(data_path)
vecs_1d = dim_reducer.fit_transform(list(df["vecs_apnews"]))
df["apnews_1"] = list(vecs_1d)

#%%


# %%
from numpy import percentile
import numpy as np
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score

seed = 42
fraction = 1.0
contamination = 0.1
df = pd.read_pickle(data_path)

X_n = int(df.shape[0] * fraction)
y_n = int(X_n * contamination)

df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
df = df[df["outlier_label"] == 1].head(X_n).append(
    df[df["outlier_label"] == -1].head(y_n))
df = df.reset_index(drop=True)

decision_scores = df["apnews_1"].astype(float).to_numpy()
threshold = percentile(decision_scores, 100 * (1 - contamination))
labels = (decision_scores > threshold)
labels = [-1 if x else 1 for x in labels]
f1_macro = f1_score(list(df["outlier_label"]), labels, average='macro')
out_f1 = f1_score(list(df["outlier_label"]), labels, pos_label=-1)
in_f1 = f1_score(list(df["outlier_label"]), labels, pos_label=1)
out_f1
#%%
unique, counts = np.unique(labels, return_counts=True)
dict(zip(unique, counts))

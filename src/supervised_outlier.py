# %%
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from evaluation import next_path
import pandas as pd
import numpy as np
from umap import UMAP
from ivis import Ivis
from evaluation import Doc2VecModel
from tqdm import tqdm
from evaluation import get_scores, reject_outliers, sample_data
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA


tqdm.pandas(desc="progess: ")

# %%


class IQROutlier:
    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X, y=None):
        pcnt = self.contamination / 2
        qlow, self.median, qhigh = np.quantile(X, [pcnt, 0.50, 1-pcnt])
        self.iqr = qhigh - qlow
        return self

    def transform(self, X, thresh_factor=1.0):
        iqr = self.iqr*thresh_factor
        preds = ((np.abs(X - self.median)) >= iqr/2)
        return [-1 if x else 1 for x in preds]


# %%
seed = 42
test_size = 0.2
labled_data = 1
outlier_class = 0
n_oe = 10000
use_ivis = True

data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
df = pd.read_pickle(data_path)
df = df[["text", "target", "outlier_label"]]
df_oe = df.where(df.target == -1).dropna()
df_oe = df_oe.iloc[np.random.RandomState(seed=seed).permutation(len(df_oe))].head(n_oe)
df_oe["label"], df_oe["outlier_label"] = 0, -1
# get all 20 news data
df = df.where(df.target != -1).dropna()
# set everything except one class to inlier
df.loc[df.target != outlier_class, "outlier_label"] = 1
# create labels for UMAP and ivis that
# are 0 and 1 (derived from the just created outlier labels)
df["label"] = (df["outlier_label"]+1)/2
# stratified sample and set unlabeled data based on labeled_data variable
df_labeled = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=1-labled_data, random_state=seed))

df = pd.merge(df, df_labeled, how='outer', indicator=True)
df.loc[df._merge == "both", "label"] = -1
df.groupby(['label','outlier_label']).size().reset_index().rename(columns={0:'count'})
#%%

df, df_test = train_test_split(df,
                               test_size=test_size, random_state=seed,
                               stratify=df["outlier_label"])
df = df.append(df_oe)
contamination = df_test["outlier_label"].value_counts(normalize=True)[-1]
print(f"df:\n {df.outlier_label.value_counts()}\ndf_test:\n {df_test.outlier_label.value_counts()}")
print(f"contamination: {contamination}")
# %%
doc2vec_path = "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin"
doc2vec_model = Doc2VecModel("apnews", "apnews", 1.0,
                             100, 1, doc2vec_path)
docvecs = doc2vec_model.vectorize(df["text"])

# %%
# UMAP
umap_n_components = 256 if use_ivis else 1
umap_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                    n_components=umap_n_components, random_state=42, 
                    verbose=True)
umap_reducer = umap_reducer.fit(list(docvecs), y=df["label"])
dim_reduced_vecs = umap_reducer.transform(list(docvecs))
if not use_ivis:
    decision_scores = dim_reduced_vecs.astype(float)

# %%
# Ivis
if use_ivis:
    ivis_reducer = Ivis(embedding_dims=1, k=15, model="maaten",
                        n_epochs_without_progress=15)
    ivis_reducer = ivis_reducer.fit(dim_reduced_vecs, Y=df["label"].to_numpy())
    dim_reduced_vecs = ivis_reducer.transform(dim_reduced_vecs)
    decision_scores = dim_reduced_vecs.astype(float)

#%%
iqrout = IQROutlier(contamination=contamination)
iqrout = iqrout.fit(decision_scores)

preds = iqrout.transform(decision_scores)
scores = get_scores(dict(), df["outlier_label"], preds)
scores

#%%
# vectorize validation texts
docvecs_test = doc2vec_model.vectorize(df_test["text"])

# umap transform validation data
dim_reduced_vecs_test = umap_reducer.transform(list(docvecs_test))
decision_scores_test = dim_reduced_vecs_test.astype(float)

if use_ivis:
    vecs_ivis_test = ivis_reducer.transform(dim_reduced_vecs_test)
    decision_scores_test = vecs_ivis_test.astype(float)

# %%
preds = iqrout.transform(decision_scores_test, thresh_factor=1)
scores = get_scores(dict(), df_test["outlier_label"], preds)
scores
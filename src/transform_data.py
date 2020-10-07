# %%
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
from sklearn.base import TransformerMixin, BaseEstimator

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

#%%
seed = 42
fraction = 0.9
contamination = 0.1

data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
df = pd.read_pickle(data_path)
print(f"df length = {df.shape[0]}")
df_test = df
df = sample_data(df, fraction, contamination, seed)
df_test = df_test.merge(df,on=["text", "outlier_label"], how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
print(f"train df length = {df.shape[0]}\ndf_test length = {df_test.shape[0]}")


# %%
doc2vec_path = "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin"
doc2vec_model = Doc2VecModel("apnews", "apnews", 1.0,
                             100, 1, doc2vec_path)
docvecs = doc2vec_model.vectorize(df["text"])

# %%
# UMAP
umap_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                   n_components=256, random_state=42)
umap_reducer = umap_reducer.fit(list(docvecs))
dim_reduced_vecs = umap_reducer.transform(list(docvecs))
#decision_scores = dim_reduced_vecs.astype(float)

# %%
# Ivis
ivis_reducer = Ivis(embedding_dims=1, k=15, model="maaten",
                   n_epochs_without_progress=15)
ivis_reducer = ivis_reducer.fit(dim_reduced_vecs)
dim_reduced_vecs = ivis_reducer.transform(dim_reduced_vecs)
decision_scores = dim_reduced_vecs.astype(float)

# %%
iqrout = IQROutlier(contamination=0.1)
iqrout = iqrout.fit(decision_scores)

preds = iqrout.transform(decision_scores)
scores = get_scores(dict(), df["outlier_label"], preds)
scores

# %%
# validate

df_val = pd.read_csv("/home/philipp/projects/dad4td/data/raw/amazon.csv", names=["stars","head","text"])
df_val["outlier_label"] = -1
df_val = df_val.iloc[np.random.RandomState(seed=42).permutation(len(df_val))]
df_val = df_val[["text", "outlier_label"]].head(10000).reset_index(drop=True)
df_val = df_val.append(df_test[["text", "outlier_label"]].where(df_test["outlier_label"]==1)).dropna().reset_index(drop=True)
df_val

#%%
df_val = df_test
#%%

# vectorize validation texts
docvecs_test = doc2vec_model.vectorize(df_val["text"])

# umap transform validation data
dim_reduced_vecs_test = umap_reducer.transform(list(docvecs_test))
#decision_scores_test = dim_reduced_vecs_test.astype(float)

vecs_ivis_test = ivis_reducer.transform(dim_reduced_vecs_test)
decision_scores_test = vecs_ivis_test.astype(float)

#%%
preds = iqrout.transform(decision_scores_test, thresh_factor=1.0)
scores = get_scores(dict(), df_val["outlier_label"], preds)
scores
# %%
from sklearn.metrics import confusion_matrix
from dotmap import DotMap
from sklearn.svm import SVC, OneClassSVM
from pyod.models.ocsvm import OCSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.engine.network import Network
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from evaluation import Doc2VecModel
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import remove_short_texts, get_scores, next_path, product_dict
import gc
import tensorflow as tf

tqdm.pandas(desc="progess: ")


# %%
df = pd.read_pickle(
    "/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/rvl_cdip.pkl")
inliers = [0, 1, 2, 11]
outliers = [4, 5, 6, 7, 9, 10, 12, 13, 14, 15]
unused_classes = [3, 8]
n_class = 20000
contamination = 0.1
min_len = 250
ref_data = "same"

df = df.where(~df.target.isin(unused_classes))
df["label"] = 0
df.loc[df.target.isin(inliers), "label"] = 1
# %%
df = df.dropna()
df = remove_short_texts(
    df=df, data_name="Full DF", min_len=250)
# %%
# get only n samples
df = df.groupby('target', group_keys=False).apply(
    lambda df: df.sample(n=min(df.shape[0], n_class), random_state=42))
df = df.reset_index(drop=True)
df.target.value_counts()

# %%
# shuffle
df = df.sample(frac=1)
# apply contamination factor
x_n = df[df.label == 1].shape[0]
df = df[df["label"] == 1].head(x_n).append(
    df[df["label"] == 0].head(int(x_n*contamination)))

df = df.reset_index(drop=True)
df.target.value_counts()

# %%
# split data into train_target, train_reference and test
df, df_test = train_test_split(df, test_size=int(df.shape[0]*0.1), random_state=42,
                               stratify=df["target"])
df_t = df.where(df.label == 1).dropna()
if ref_data == "same":
    df_r = df.where(df.label == 0).dropna()
elif ref_data == "20_news":
    df_r = pd.read_pickle(
        "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")
    df_r = df_r.where(df_r.target != -1).dropna()
elif ref_data == "both":
    df_r = df.where(df.label == 0).dropna()
    df_20 = pd.read_pickle(
        "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")
    df_20 = df_20.where(df_20.target != -1).dropna()
    df_20.target = (df_20.target + 1)*20
    df_r = df_r.append(df_20)
else:
    raise Exception(
        f"{ref_data} not valid value for ref_data. Must be one of: same, 20_news")
df_r
# %%
df_r.target.value_counts()
# %%
print("df_t\n", df_t.target.value_counts())
print("df_r\n", df_r.target.value_counts())
print("df_test\n", df_test.target.value_counts())
# %%

doc2vecwikiall = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                              100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")
print("get train target vecs")
df_t["vecs"] = doc2vecwikiall.vectorize(df_t["text"])
print("get train reference vecs")
df_r["vecs"] = doc2vecwikiall.vectorize(df_r["text"])
print("get test vecs")
df_test["vecs"] = doc2vecwikiall.vectorize(df_test["text"])
#%%
from umap import UMAP
from one_class import prep_data

# config
mode = "one_out"
c = DotMap()
c.weakly = [None]
c.batchsize = [128]
c.epoch_num = [12]
c.epoch_report = [4]
c.sup_epochs = [15]
c.feature_out = [64]
c.pred_mode = ["ocsvm"]
c.threshold = [0.55]
c.n = [100, 1000, 10000]
c.random_state = range(1,6)

c = [DotMap(x) for x in product_dict(**c)]


def ocsvm_train_test(result_df, df_t, df_r, df_test, res_path, c):
    # data
    x_target = np.array(df_t.sample(n=c.n).vecs.to_list())
    test_vecs = np.array(df_test.vecs.to_list())
    labels = df_test["label"].astype(int).values

    clf = OneClassSVM()
    clf.fit(x_target)

    preds = clf.predict(test_vecs)
    decision_scores = preds

    scores = get_scores(labels, preds, outlabel=0, threshold=c.threshold)
    print(f"\n\nTest scores:\n{pd.DataFrame([scores], index=[0])}")
    normalize="true"
    print(f"{confusion_matrix(labels, np.where(decision_scores > c.threshold, 1, 0), normalize=normalize)}")


    result_df = result_df.append(dict(cclass=list(
        df_test.target.unique()), **scores, **c), ignore_index=True)
    result_df.to_csv(res_path, sep="\t")

    return result_df

# %%
i=0
result_df = pd.DataFrame()

res_path = next_path(
        "/home/philipp/projects/dad4td/reports/one_class/one_out_%04d_pure_ocsvm.tsv")
for i in outliers:
    print(f"class in test: {i}")
    for params in c:
        print(params)
        result_df = ocsvm_train_test(result_df, *prep_data([i], df_t, df_r, df_test, weakly=params.weakly, ref_data =ref_data, only_outlier_in_test=False), 
            res_path=res_path, c=params)
# %%
res_path = next_path(
        "/home/philipp/projects/dad4td/reports/one_class/all_%04d_pure_ocsvm.tsv")
remap = {k: v for v, k in zip(
    range(df_r.target.unique().shape[0]), df_r.target.unique())}
df_r_temp = df_r.copy(deep=True)
df_r_temp.target = df_r_temp.target.map(remap)
for params in c:
    print(params)
    result_df = ocsvm_train_test(result_df, df_t, df_r_temp, df_test, res_path=res_path, c=params)
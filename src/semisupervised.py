# %%
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from evaluation import next_path
import pandas as pd
import numpy as np
from umap import UMAP
from ivis import Ivis
from evaluation import Doc2VecModel, product_dict
from tqdm import tqdm
from evaluation import get_scores, reject_outliers, sample_data
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from itertools import permutations


tqdm.pandas(desc="progess: ")


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


def get_outlier_data(oe_path, n_oe, seed):
    df_oe = pd.read_pickle(oe_path)
    df_oe = df_oe.iloc[np.random.RandomState(
        seed=seed).permutation(len(df_oe))].head(n_oe)
    df_oe["label"], df_oe["outlier_label"], df_oe["scorable"] = 0, -1, 0
    return df_oe


def label_data(df, seed, labeled_data, outlier_classes):
    df = df[["text", "target", "vecs"]]
    df["scorable"] = 1
    # get all 20 news data
    df = df.where(df.target != -1).dropna()
    # set everything except one class to inlier
    df["outlier_label"] = -1
    df.loc[~df.target.isin(outlier_classes), "outlier_label"] = 1
    # create labels for UMAP and ivis that
    # are 0 and 1 (derived from the just created outlier labels)
    df["label"] = (df["outlier_label"]+1)/2
    # stratified sample and set unlabeled data based on labeled_data variable
    df_unlabeled = df[["text", "outlier_label"]].groupby('outlier_label', group_keys=False).apply(
        lambda x: x.sample(frac=1-labeled_data, random_state=seed)).reset_index(drop=True)

    df = pd.merge(df.reset_index(drop=True), df_unlabeled.reset_index(
        drop=True), how='outer', indicator=True)
    df = df.drop_duplicates()

    df.loc[df._merge == "both", "label"] = -1
    print("Data before split:\n")
    print(df.groupby(['label', 'outlier_label']).size(
    ).reset_index().rename(columns={0: 'count'}), "\n")
    return df


def prepare_data(df, outliers, inliers, seed, fixed_cont, labeled_data, n_oe, test_size, **kwargs):
    df = df_full.where(df_full.target.isin(
        outliers+inliers)).dropna()
    # label data as inliers and outliers (for scoring) and whether
    # they have labels or not (semi-supervised)
    df = label_data(df, seed, labeled_data, outliers)

    if fixed_cont:
        df = sample_data(df, 1.0, fixed_cont, seed)
        print("Data after adjusting for fixed contamination:\n")
        print(df.groupby(['label', 'outlier_label']).size(
        ).reset_index().rename(columns={0: 'count'}), "\n")

    if n_oe:
        df_oe = get_outlier_data(oe_path, n_oe, seed)
        df_oe["vecs"] = doc2vec_model.vectorize(df_oe["text"])

    contamination = df.outlier_label.value_counts(normalize=True)[-1]
    params["contamination"] = contamination
    print(f"Contamination: {contamination}\n")

    # split train test
    df, df_test = train_test_split(df,
                                   test_size=test_size, random_state=seed,
                                   stratify=df["outlier_label"])
    if n_oe:
        df = df.append(df_oe)

    if -1 in df.label.unique() and df.label.value_counts()[-1] != df.shape[0]:
        if df[(df.label == 0) & (df.outlier_label == -1)].shape[0] == 0:
            print("Adding missing sample for labeled outlier")
            df.loc[((df.label == -1) & (df.outlier_label == -1)).idxmax(), 'label'] = 0


    print("Training data:\n",df.groupby(['label', 'outlier_label']).size(
    ).reset_index().rename(columns={0: 'count'}), "\n\n")
    print("Test data:\n",df_test.groupby(['label', 'outlier_label']).size(
    ).reset_index().rename(columns={0: 'count'}), "\n\n")
    
    return df, df_test


def umap_reduce(docvecs, label, umap_model, use_ivis, **kwargs):
    if not umap_model:
        print(f"Train UMAP...")
        umap_n_components = min(256, len(docvecs)-2) if use_ivis else 1
        umap_model = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                          n_components=umap_n_components, random_state=42,
                          verbose=False)
        umap_model = umap_model.fit(docvecs, y=label)
    dim_reduced_vecs = umap_model.transform(docvecs)
    if not use_ivis:
        dim_reduced_vecs = dim_reduced_vecs.astype(float)
    return dim_reduced_vecs, umap_model


def ivis_reduce(docvecs, label, ivis_model, use_ivis, **kwargs):
    if use_ivis:
        if not ivis_model:
            print(f"Train ivis...")
            ivis_model = Ivis(embedding_dims=1, k=15, model="maaten",
                              n_epochs_without_progress=15, verbose=0,
                              batch_size=max(1, min(128, df_test.shape[0]-1)))
            if -1 in label.unique() and label.value_counts()[-1] == label.shape[0]:
                print("No labeled data found.")
                ivis_model = ivis_model.fit(docvecs)
            else:
                ivis_model = ivis_model.fit(
                    docvecs, Y=label.to_numpy())

        dim_reduced_vecs = ivis_model.transform(docvecs)
        decision_scores = dim_reduced_vecs.astype(float)
        return decision_scores, ivis_model
    else:
        return docvecs, None


def score_out_preds(docvecs, iqr_model, contamination, **kwargs):
    if not iqr_model:
        iqrout = IQROutlier(contamination=contamination)
        iqrout = iqrout.fit(docvecs)
    preds = iqrout.transform(docvecs)
    return preds, iqrout


standard_split = [([0, 1, 2, 11], [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15])]
pairwise_split = list(permutations([[x] for x in range(0, 16)], 2))
# %%
param_combinations = product_dict(**dict(
    seed=[42, 43, 44],
    test_size=[0.2],
    labeled_data=[0.1, 0.3,  0.5],
    fixed_cont=[0.05, 0.1],
    n_oe=[0],
    use_ivis=[True],
    pair=pairwise_split
))
# how many samples per class are used for all tests
n_class = 500

# split the outlier, inlier tuple pairs and print all parameters for run
for d in param_combinations:
    d["inliers"], d["outliers"] = d["pair"]
    d.pop('pair', None)

#data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
data_path = "/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/rvl_cdip.pkl"
oe_path = "/home/philipp/projects/dad4td/data/processed/oe_data.pkl"
res_path = next_path(
    "/home/philipp/projects/dad4td/reports/semisupervised/semisup_rvl_pw_%04d.tsv")

doc2vec_model = Doc2VecModel("apnews", "apnews", 1.0,
                             100, 1,
                             "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin")

# load data and get the doc2vec vectors for all of the data used
df_full = pd.read_pickle(data_path)

# sample only a portion of the data
df_full = df_full.groupby('target', group_keys=False).apply(
    lambda df: df.sample(n=n_class, random_state=42))

# %%
df_full["vecs"] = doc2vec_model.vectorize(df_full["text"])
df_full["vecs"] = df_full["vecs"].apply(tuple)

# %%
result_df = pd.DataFrame()
for i, params in enumerate(param_combinations):
    print(
        f"\n\n---------------------\n\nRun {i+1} out of {len(param_combinations)}\n\n{params}")

    df, df_test = prepare_data(df_full, **params)

    # UMAP Train
    docvecs, umap_model = umap_reduce(
        df["vecs"].to_list(), df["label"], None, **params)

    # Ivis
    docvecs, ivis_model = ivis_reduce(
        docvecs, df["label"], None, **params)

    # remove OE data, so it's not scored as well
    df["decision_scores"] = docvecs
    df = df.where(df.scorable == 1).dropna()

    # find outliers in 1D scores
    preds, iqr_model = score_out_preds(df["decision_scores"], None, **params)

    # score the predictions for outliers
    scores = get_scores(dict(), df["outlier_label"], preds)

    # %%
    #  write the scores to df and save
    scores.update(params)
    scores["data"] = "train"
    result_df = result_df.append(scores, ignore_index=True)
    result_df.to_csv(res_path, sep="\t")
    print(f"\nTraining scores:\n{pd.DataFrame([scores], index=[0])}")
    # %%
    # test UMAP and ivis
    docvecs_test, _ = umap_reduce(
        df_test["vecs"].to_list(), None, umap_model, **params)

    docvecs_test, _ = ivis_reduce(docvecs_test, None, ivis_model, **params)

    # remove OE data, so it's not scored as well
    df_test["decision_scores"] = docvecs_test
    df_test = df_test.where(df_test.scorable == 1).dropna()

    # find outliers in 1D scores
    preds = iqr_model.transform(df_test["decision_scores"], thresh_factor=1)

    # score the predictions for outliers
    scores = get_scores(dict(), df_test["outlier_label"], preds)

    # write the scores to df and save
    scores.update(params)
    scores["data"] = "test"
    result_df = result_df.append(scores, ignore_index=True)
    result_df.to_csv(res_path, sep="\t")
    print(f"\nTest scores:\n{pd.DataFrame([scores], index=[0])}")

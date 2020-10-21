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


def get_outlier_data(oe_path, n_oe):
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


seeds = [42, 43, 44]
test_size = 0.2
labeled_data = 1.0
outlier_classes = None
fixed_cont = 0.2
n_oe = 0
use_ivis = True

#data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
data_path = "/home/philipp/projects/dad4td/data/processed/rvl_cdip.pkl"
oe_path = "/home/philipp/projects/dad4td/data/processed/oe_data.pkl"
res_path = next_path(
    "/home/philipp/projects/dad4td/reports/sup_combs_rvl_%04d.tsv")
doc2vec_model = Doc2VecModel("apnews", "apnews", 1.0,
                             100, 1,
                             "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin")
df_full = pd.read_pickle(data_path)
# get the doc2vec vectors for all of the data used
df_full["vecs"] = doc2vec_model.vectorize(df_full["text"])
df_full["vecs"] = df_full["vecs"].apply(tuple)

pairs = list(permutations(range(0, 16), 2))
runs = len(pairs)
result_df = pd.DataFrame()
for i, (inlier, outlier) in enumerate(pairs):
    for seed in seeds:
        outlier_classes, inlier_classes = [outlier], [inlier]
        params = dict(seed=seed, test_size=test_size, labeled_data=labeled_data,
                    outlier_classes=outlier_classes, inlier_classes=inlier_classes,
                    n_oe=n_oe, use_ivis=use_ivis)
        print(f"\n\n---------------------\n\nRun {i+1} out of {runs}\n\n{params}")

        df = df_full.where(df_full.target.isin(
            outlier_classes+inlier_classes)).dropna()
        # label data as inliers and outliers (for scoring) and whether
        # they have labels or not (semi-supervised)
        df = label_data(df, seed, labeled_data, outlier_classes)

        if fixed_cont:
            df = sample_data(df, 1.0, fixed_cont, seed)
            print("Data after adjusting for fixed contamination:\n")
            print(df.groupby(['label', 'outlier_label']).size(
            ).reset_index().rename(columns={0: 'count'}), "\n")

        if n_oe:
            df_oe = get_outlier_data(oe_path, n_oe)
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

        print(
            f"Training data:\n {df.outlier_label.value_counts()}\n\nTest data:\n {df_test.outlier_label.value_counts()}")

        # %%
        # UMAP
        print(f"Train UMAP...")
        docvecs = df["vecs"].to_list()
        umap_n_components = min(256, len(docvecs)-2) if use_ivis else 1
        umap_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                            n_components=umap_n_components, random_state=42,
                            verbose=False)
        umap_reducer = umap_reducer.fit(docvecs, y=df["label"])
        dim_reduced_vecs = umap_reducer.transform(docvecs)
        if not use_ivis:
            decision_scores = dim_reduced_vecs.astype(float)

        # %%
        # Ivis
        if use_ivis:
            print(f"Train ivis...")
            ivis_reducer = Ivis(embedding_dims=1, k=15, model="maaten",
                                n_epochs_without_progress=15, verbose=0,
                                batch_size=min(128, df_test.shape[0]-1))
            ivis_reducer = ivis_reducer.fit(
                dim_reduced_vecs, Y=df["label"].to_numpy())
            dim_reduced_vecs = ivis_reducer.transform(dim_reduced_vecs)
            decision_scores = dim_reduced_vecs.astype(float)

        # %%
        df["decision_scores"] = decision_scores
        df = df.where(df.scorable == 1).dropna()
        # %%
        iqrout = IQROutlier(contamination=contamination)
        iqrout = iqrout.fit(df["decision_scores"])

        preds = iqrout.transform(df["decision_scores"])
        scores = get_scores(dict(), df["outlier_label"], preds)

        scores.update(params)
        scores["data"] = "train"
        result_df = result_df.append(scores, ignore_index=True)
        result_df.to_csv(res_path, sep="\t")
        print(f"\nTraining scores:\n{pd.DataFrame(scores, index=[0])}")

        # %%
        docvecs_test = df_test["vecs"].to_list()
        # umap transform validation data
        dim_reduced_vecs_test = umap_reducer.transform(list(docvecs_test))
        decision_scores_test = dim_reduced_vecs_test.astype(float)

        if use_ivis:
            vecs_ivis_test = ivis_reducer.transform(dim_reduced_vecs_test)
            decision_scores_test = vecs_ivis_test.astype(float)
        # %%
        df_test["decision_scores"] = decision_scores_test
        df_test = df_test.where(df_test.scorable == 1).dropna()
        # %%

        preds = iqrout.transform(df_test["decision_scores"], thresh_factor=1)
        scores = get_scores(dict(), df_test["outlier_label"], preds)
        scores.update(params)
        scores["data"] = "test"
        result_df = result_df.append(scores, ignore_index=True)
        result_df.to_csv(res_path, sep="\t")
        print(f"\nTest scores:\n{pd.DataFrame(scores, index=[0])}")
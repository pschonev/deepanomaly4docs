# %%
from keras import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from utils import next_path, product_dict, get_scores, reject_outliers, sample_data, remove_short_texts
import pandas as pd
import numpy as np
from umap import UMAP
from ivis import Ivis
from evaluation import Doc2VecModel, TransformerModel
from tqdm import tqdm
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from itertools import permutations
from semisupervised import prepare_data, umap_reduce, get_outlier_data

tqdm.pandas(desc="progess: ")
# %%

# %%
def get_weakly_data(df, weakly_supervised, in_test_not_train_outlier, seed, **kwargs):
    df = df.where(df.target.isin(in_test_not_train_outlier)).dropna()
    df = df.groupby(df.target).apply(lambda d: d.sample(
        n=weakly_supervised, random_state=seed))
    df["outlier_label"] = -1
    df["label"] = 0
    return df


def create_model(n_out=16, loss="categorical_crossentropy", dropout_rate=0.2, n_in=256):
    model = Sequential()
    model.add(Dense(128, input_dim=n_in, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_out, activation='sigmoid'))

    model.compile(loss=loss,
                  optimizer='adam', metrics=['accuracy'])
    return model


def neuralnet(docvecs, label, model, n_out, loss, use_nn, epochs, **kwargs):
    if use_nn:
        if not model:
            print(f"Train NN...")
            model = create_model(n_out=n_out, loss=loss, n_in=docvecs.shape[1])
            model.fit(
                docvecs, y=label, epochs=epochs, batch_size=64, verbose=1)
        dim_reduced_vecs = model.predict(docvecs)
        decision_scores = dim_reduced_vecs.astype(float)
        return decision_scores, model
    else:
        return docvecs, model




# unused classes 3 handwritten and 8 file folder
inliers = [0, 1, 2, 11]
outliers = [4, 5, 6, 7, 9, 10, 12, 13, 14, 15]
unused_classes = [3, 8]

standard_split = [(inliers, outliers, inliers, outliers)]
# test pairwise inlier/outlier for each combination of classes
pairwise_split = [(x[0], x[1], x[0], x[1]) for x in list(
    permutations([[x] for x in range(0, 16) if x not in unused_classes], 2))]
# normal train/test split but one unseen outlier
one_new_outlier = [(inliers, [j for j in outliers if j != i], [], [
                    j for j in outliers if j == i]) for i in outliers]
one_new_outlier_w_inlier = [(inliers, [j for j in outliers if j != i], inliers, [
    j for j in outliers if j == i]) for i in outliers]

o = 7
one_new_outlier_w_inlier_i = [(inliers, [j for j in outliers if j != o],
                               inliers, [o])]

one_to_many = [(inliers, [j for j in outliers if j == i], [], [
                j for j in outliers if j != i]) for i in outliers]

one_to_many_w_inlier = [(inliers, [j for j in outliers if j == i], inliers, [
                j for j in outliers if j != i]) for i in outliers]

# how many samples per class are used for all tests
n_classes = [20000]
test_size = 3000
contaminations = [0.1]
n_oe=[False]

# %%
param_combinations = product_dict(**dict(
    seed=range(1, 4),
    labeled_data=[1.0],
    fixed_cont=contaminations,
    n_oe=n_oe,
    use_nn=[True],
    use_umap=[False],
    min_len=[200],
    epochs=[15],
    class_split=one_new_outlier_w_inlier_i,
    weakly_supervised=[False],
    balanced=[False]
))

# split the outlier, inlier tuple pairs and print all parameters for run
for d in param_combinations:
    d["inliers"], d["outliers"], d["test_inliers"], d["test_outliers"] = d["class_split"]
    d.pop('pair', None)
    d["in_test_not_train_outlier"] = [
        x for x in d["test_outliers"] if x not in d["outliers"]]
print(param_combinations)

#data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
data_path = "/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/rvl_cdip.pkl"
oe_path = "/home/philipp/projects/dad4td/data/processed/oe_data.pkl"
res_path = next_path(
    "/home/philipp/projects/dad4td/reports/supervised/sup_one_to_many_UMAP%04d.tsv")

save_best = False
best_pred_path = next_path(
    "/home/philipp/projects/dad4td/reports/supervised/sup_one_to_many%04d.tsv")


####
# data preparation
####

# load data and get the doc2vec vectors for all of the data used
df_full = pd.read_pickle(data_path)

# remove unused classes
df_full = df_full[~df_full.target.isin(unused_classes)]


# split train test
df_full, df_test_full = train_test_split(df_full, test_size=test_size, random_state=42,
                                         stratify=df_full["target"])

# %%
# only take as many samples as needed at most
df_full = df_full.groupby('target', group_keys=False).apply(
    lambda df: df.sample(n=min(df.shape[0], max(n_classes)), random_state=42))
df_full_in = df_full.where(df_full.target.isin(inliers)).dropna()
df_full_out = df_full.where(df_full.target.isin(outliers)).sample(
        frac=max(contaminations), random_state=42).dropna()
df_full = df_full_in.append(df_full_out)
df_full.target.value_counts()
# %%
if max(n_oe):
    df_oe = get_outlier_data(oe_path, max(n_oe), 42)
    print(f"df_oe: {df_oe.shape[0]}")
else:
    df_oe = None
# %%
# vectorization model
#longformer_base = TransformerModel(
#    "allenai/longformer-base-4096", "long_documents", 149)
doc2vecwikiall = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                              100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")


doc2vec_models = [doc2vecwikiall]
doc2vec_model = doc2vecwikiall

result_df = pd.DataFrame()
best_f1 = 0
i = 0


print(f" Running model: {doc2vec_model.model_name}")
print(f" Train data ...")
df_full["vecs"] = doc2vec_model.vectorize(df_full["text"])
df_full["vecs"] = df_full["vecs"].apply(tuple)
print(f" OE data ...")
if max(n_oe):
    df_oe["vecs"] = doc2vec_model.vectorize(df_oe["text"])
    df_oe["vecs"] = df_oe["vecs"].apply(tuple)
print(f" Test data ...")
df_test_full["vecs"] = doc2vec_model.vectorize(df_test_full["text"])
df_test_full["vecs"] = df_test_full["vecs"].apply(tuple)

# %%

for n_class in n_classes:
    # sample only a portion of the data
    df_partial = df_full.groupby('target', group_keys=False).apply(
        lambda df: df.sample(n=min(df.shape[0], n_class), random_state=42))

    for params in param_combinations:
        print(
            f"\n\n---------------------\n\nRun {i+1} out of {len(param_combinations)*len(n_classes)*len(doc2vec_models)}\n\n{params}")
        print(f"model: {doc2vec_model.model_name}")
        print(f"n_class: {n_class}")

        # remove short strings
        if params["min_len"] is not False:
            df = remove_short_texts(
                df=df_partial, data_name="Train DF", min_len=params["min_len"])
            df_test = remove_short_texts(
                df=df_test_full, data_name="Test DF", min_len=params["min_len"])
        else:
            df = df_partial
            df_test = df_test_full

        # weakly supervised classes
        if params["weakly_supervised"]:
            df_weakly = get_weakly_data(df, **params)
        # label the trainign data
        df = prepare_data(
            df, doc2vec_model=doc2vec_model, df_oe=df_oe, **params)
        # combine
        if params["weakly_supervised"]:
            df = df.append(df_weakly).reset_index(drop=True)

        # label test set
        df_test["label"] = 0
        df_test.loc[~df_test.target.isin(
            params["test_outliers"]), "label"] = 1
        df_test["outlier_label"] = -1
        df_test.loc[~df_test.target.isin(
            params["test_outliers"]), "outlier_label"] = 1

        if params["balanced"]:
            df_test_out = df_test.where(
            (df_test.label == 0) & (df_test.target.isin(params["test_outliers"]))).dropna()
            n=df_test_out.shape[0]
            df_test_in = df_test.where((df_test.label == 1) & (df_test.target.isin(params["test_inliers"])))
            df_test_in = df_test_in.dropna(how="all").sample(
                n=n, random_state=params["seed"])
            df_test = df_test_out.append(df_test_in).dropna(how="all")
        else:
            # sampling the df_test set
            df_test = sample_data(df_test, 1.0, 0.1, 42)
            df_test = df_test[df_test.target.isin(
                params["test_outliers"]+params["test_inliers"])]

        print("df_train")
        print(df.label.value_counts())
        print(df.target.value_counts())

        print("df_test")
        print(df_test.label.value_counts())
        print(df_test.target.value_counts())

        #####
        # train
        #####
        # UMAP Train
        docvecs, umap_model = umap_reduce(
            df["vecs"].to_list(), df["label"], None, **params)

        # neural net train
        label_inputs = df["label"].astype(int).reset_index(drop=True)
        _, nnet = neuralnet(
            docvecs, label_inputs, None, n_out=1, loss="binary_crossentropy", **params)

        #####
        # test
        #####
        # test UMAP and neural net
        docvecs_test, _ = umap_reduce(
            df_test["vecs"].to_list(), None, umap_model, **params)

        docvecs_test, model = neuralnet(
            docvecs_test, None, nnet, n_out=1, loss="binary_crossentropy", **params)

        # get prediction scores
        threshold = 0.5
        scores = get_scores(df_test["label"].astype(
            int).values, docvecs_test, outlabel=0)

        ####
        # write scores
        ####
        # write the scores to df and save
        scores.update(params)
        scores["n_class"] = n_class
        scores["data"] = "test"
        scores["threshold"] = threshold
        scores["doc2vec_model"] = doc2vec_model.model_name
        result_df = result_df.append(scores, ignore_index=True)
        result_df.to_csv(res_path, sep="\t")
        print(f"\nTest scores:\n{pd.DataFrame([scores], index=[0])}")

        if scores["f1_macro"] > best_f1 and save_best:
            best_f1 = scores["f1_macro"]
            df_best_pred = df_test
            df_best_pred["pred"] = docvecs_test
            df_best_pred.to_csv(best_pred_path, sep="\t")
        i += 1

# %%
umap_model = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                  n_components=2, random_state=42,
                  verbose=False)

features_doc2vec = df_test["vecs"].to_list()
features_doc2vec = umap_model.fit_transform(features_doc2vec)
features_doc2vec

# %%
norm_doc2vec_2d = normalize(features_doc2vec, norm="l1")
norm_doc2vec_2d
# %%


def load_coords_to_df(df, coords_2d):
    df["X"] = coords_2d[:, 0]
    df["Y"] = coords_2d[:, 1]

    return df


df_test = load_coords_to_df(df_test, features_doc2vec)


sns.scatterplot('X', 'Y', data=df_test, hue="label")
plt.title('Doc2Vec - UMAP')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# %%

features = Model(inputs=model.input, outputs=model.layers[-2].output)
features.summary()

test_vecs = df_test["vecs"].to_list()
test_vecs = np.array(test_vecs).astype(float)
feature_vec = features.predict(test_vecs)
feature_vec

# %%
feature_vec_2d = umap_model.fit_transform(feature_vec)
df_test = load_coords_to_df(df_test, feature_vec_2d)

sns.scatterplot('X', 'Y', data=df_test, hue="label")
plt.title('Fully Supervised')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# %%
df_test

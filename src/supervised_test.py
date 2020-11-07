# %%
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential
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
from semisupervised import IQROutlier, prepare_data, umap_reduce

tqdm.pandas(desc="progess: ")

standard_split = [([0, 1, 2, 11], [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15])]
pairwise_split = list(permutations([[x] for x in range(0, 16)], 2))
# %%
param_combinations = product_dict(**dict(
    seed=[42],
    test_size=[0.2],
    labeled_data=[1.0],
    fixed_cont=[0.1],
    n_oe=[0],
    use_ivis=[True],
    pair=standard_split
))

# split the outlier, inlier tuple pairs and print all parameters for run
for d in param_combinations:
    d["inliers"], d["outliers"] = d["pair"]
    d.pop('pair', None)

#data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
data_path = "/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/rvl_cdip.pkl"
oe_path = "/home/philipp/projects/dad4td/data/processed/oe_data.pkl"
res_path = next_path(
    "/home/philipp/projects/dad4td/reports/sup_combs_rvl_%04d.tsv")

# how many samples per class are used for all tests
n_class = 1000

# %%
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
df, df_test = prepare_data(df_full, [0, 1, 2, 11], [
                           3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15], seed=42,
                           fixed_cont=0.1, labeled_data=1.0, n_oe=0, test_size=0.2)

# UMAP Train
docvecs, umap_model = umap_reduce(
    df["vecs"].to_list(), df["target"], None, True)


# %%
classes = False
label = "target" if classes else "label"
n_out = 16 if classes else 1
loss = "categorical_crossentropy" if classes else "binary_crossentropy"


def create_model(n_out=16, loss="categorical_crossentropy", dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(128, input_dim=256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_out, activation='sigmoid'))

    model.compile(loss=loss,
                  optimizer='adam', metrics=['accuracy'])
    return model


def neuralnet(docvecs, label, model, use_ivis, **kwargs):
    if use_ivis:
        if not model:
            print(f"Train NN...")
            model = create_model(n_out=n_out, loss=loss)
            model.fit(
                docvecs, y=label, epochs=15, batch_size=64, verbose=1)
        dim_reduced_vecs = model.predict(docvecs)
        decision_scores = dim_reduced_vecs.astype(float)
        return decision_scores, model
    else:
        return docvecs, None


# NN
if classes:
    label_inputs = to_categorical(df[label].astype(int).reset_index(drop=True))
else:
    label_inputs = df[label].astype(int).reset_index(drop=True)
docvecs_after_nn, nnet = neuralnet(
    docvecs, label_inputs, None, True)


np.unique(docvecs_after_nn, return_counts=True)
# %%
# test UMAP and ivis
docvecs_test, _ = umap_reduce(
    df_test["vecs"].to_list(), None, umap_model, True)

docvecs_test, _ = neuralnet(docvecs_test, None, nnet, True)
# %%
if classes:
    print(pd.DataFrame(np.unique(np.argmax(docvecs_test, axis=-1), return_counts=True)))
else:
    threshold = 0.5
    out_df = pd.DataFrame(
        np.unique(np.where(docvecs_test > threshold, 1, 0), return_counts=True))
    print(tabulate(out_df,
                   out_df.columns, tablefmt="rst"))

# %%
df_test["label"].value_counts()
# %%

if classes:
    scores = get_scores(dict(), df_test[label].astype(
        int).values, np.argmax(docvecs_test, axis=-1))
else:
    scores = get_scores(dict(), df_test[label].astype(
        int).values, np.where(docvecs_test > threshold, 1, 0), outlabel=0)
out_df = pd.DataFrame(scores, index=[0]).round(3)
print(tabulate(out_df,
               out_df.columns, tablefmt="rst"))

# %%
plt.hist(docvecs_test, bins=30)
fig = plt.figure(1)
fig.set_facecolor("white")
# plt.savefig("/home/philipp/projects/dad4td/reports/semisupervised/distribution_output_supervised.png",
#            facecolor=fig.get_facecolor(), dpi=400)
# %%
plot_model(ivis_model,
           to_file='/home/philipp/projects/dad4td/reports/semisupervised/model_plot.png',
           show_shapes=True, show_layer_names=True)
# %%
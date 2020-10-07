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

tqdm.pandas(desc="progess: ")
#%%

seed = 42
fraction = 1.0
contamination = 0.1

data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
scores = dict()

df = pd.read_pickle(data_path)
df = sample_data(df, fraction, contamination, seed)

df.columns

# %%
doc2vec_path = "/home/philipp/projects/dad4td/models/all_news_05_30_30/all_news.bin"
doc2vec_model = Doc2VecModel("all_news_05_30_30", "all_news", 0.5,
                             30, 30, doc2vec_path)
docvecs = doc2vec_model.vectorize(df["text"])

# %%
# UMAP
dim_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                   n_components=256, random_state=42)

dim_reduced_vecs = dim_reducer.fit_transform(list(docvecs))
decision_scores = dim_reduced_vecs.astype(float)

# %%
# Ivis
dim_reducer = Ivis(embedding_dims=2, k=15, model="maaten",
                   n_epochs_without_progress=15)
docvecs = np.vstack(df["apnews_256"].to_numpy())
dim_reduced_vecs = dim_reducer.fit_transform(docvecs)
#decision_scores = dim_reduced_vecs.astype(float)

#%%
od = OCSVM(verbose=True)
od.fit(dim_reduced_vecs)

out_pred = od.labels_
out_pred[out_pred == 1] = -1
out_pred[out_pred == 0] = 1

scores = get_scores(dict(), df["outlier_label"], out_pred)
scores

# %%
# Read saved from DF
df = pd.read_pickle(data_path)
df = sample_data(df, fraction, contamination, 44)
decision_scores = df["apnews_1"].astype(float).to_numpy()

# %%
# Get outlier score

preds = reject_outliers(decision_scores, iq_range=1.0-contamination)
preds = [-1 if x else 1 for x in preds]

scores = get_scores(scores, df["outlier_label"], preds)
scores

# %%
dim_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                   n_components=256, random_state=42)

docvecs = dim_reducer.fit_transform(list(df["vecs_apnews"]))
docvecs
# %%
docvecs

# %%

seeds = [42, 43, 44]
contaminations = [0.01, 0.03, 0.05, 0.1, 0.15]
data_seed = 42
fraction = 1.0
i_umap = 2
i_ivis = 2

data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
save_path = next_path(
    "/home/philipp/projects/dad4td/reports/contaminations_%04d.tsv")

print(f"Load data...")
df = pd.read_pickle(data_path)
df = sample_data(df, fraction, contamination, data_seed)

results = []
n = 1
for c, contamination in enumerate(contaminations):
    print(f"Contanimation {contamination} ({c+1} out of {len(contaminations)} contanimations)...")
    for s, seed in enumerate(seeds):
        for i in range(i_umap):
            print(
                f"Start UMAP run {i+1 } out of {i_umap} for seed {seed} ({s+1} out of {len(seeds)} seeds)...")
            start = timer()
            dim_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                            n_components=256, random_state=seed)
            docvecs = dim_reducer.fit_transform(list(df["vecs_apnews"]))
            end = timer()
            time_umap = end - start
            print(f"UMAP done in {time_umap:.2f}s.")
            for j in range(i_ivis):
                print(f"Start ivis {j+1} out of {i_ivis}...")
                start = timer()
                dim_reducer = Ivis(embedding_dims=1, k=15,
                                model="maaten", n_epochs_without_progress=10, verbose=0)
                dim_reduced_vecs = dim_reducer.fit_transform(docvecs)
                decision_scores = dim_reduced_vecs.astype(float)
                end = timer()
                time_ivis = end - start
                print(f"ivis done in {time_ivis:.2f}s.")

                preds = reject_outliers(
                    decision_scores, iq_range=1.0-contamination)
                preds = [-1 if x else 1 for x in preds]
                scores = get_scores(dict(), df["outlier_label"], preds)
                scores.update(dict(seed=seed, i_umap=i, i_ivis=j,
                                contamination=contamination, fraction=fraction,
                                time_umap=time_umap, time_ivis=time_ivis))

                print(f"Run {n} out of {len(contaminations)*len(seeds)*i_umap*i_ivis} done:")
                print(f"{pd.DataFrame(scores, index=[0])}")
                results.append(scores)
                df_results = pd.DataFrame(results)
                df_results.to_csv(save_path, sep="\t")
                n += 1

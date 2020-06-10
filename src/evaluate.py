# %%
import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline


def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = '%03d-results.tsv':

    001-results.tsv
    001-results.tsv
    001-results.tsv

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while Path(path_pattern % i).exists():
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if Path(path_pattern % c).exists() else (a, c)

    return path_pattern % b


def sample_data(df, data_frac, contamination, seed):
    X_n = int(df.shape[0] * data_frac)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    return df

# parameters
data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"
result_folder = "/home/philipp/projects/dad4td/reports/"
results_path = next_path(result_folder + "%04d_dens_eval.tsv")
results_param_path = result_folder + Path(results_path).stem + ".txt"

d = dict(data_frac=0.15,
         contamination=0.1,
         seed=42)

# prepare data
df = pd.read_pickle(data_path)
df = sample_data(df, **d)

X = df["text"]
y = df["outlier_label"]

# prepare pipeline
pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('vectorize', TfidfVectorizer(stop_words='english')),
    ('reduce_dim', 'passthrough'),
    ('classify', LocalOutlierFactor(novelty=True, contamination=d["contamination"]))
])

MIN_DF = [25] 
MIX_RATIO = [0.0, 0.1]
N_COMPONENTS = [2, 50, 300]
UMAP_METRICS = ['manhattan', 'euclidean']
LOF_METRICS = ['euclidean']

param_grid = [
    {
        'vectorize__min_df': MIN_DF,
        'reduce_dim': [UMAP(random_state=42)],
        'reduce_dim__n_components': N_COMPONENTS,
        'reduce_dim__set_op_mix_ratio': MIX_RATIO,
        'reduce_dim__metric': UMAP_METRICS,
        'classify__metric': LOF_METRICS
    }
]

# grid search
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=d["seed"])
grid = GridSearchCV(pipe, scoring='f1', param_grid=param_grid,
                    cv=cv, verbose=10, n_jobs=-1)
grid.fit(X, y)

# get the results dataframe and add all parameters outside the pipeline
out_df = pd.DataFrame.from_dict(grid.cv_results_)
for key, val in d.items():
    out_df[key] = val
out_df = out_df.sort_values(by=['rank_test_score'])
out_df.to_csv(results_path, sep="\t")

# save parameters of eval run to txt file
with open(results_param_path, 'w') as res_params:
    res_params.write(f""" {Path(results_param_path).stem}

    {data_path}

    {d}

    {grid}
    """)

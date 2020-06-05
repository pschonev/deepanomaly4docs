# %%
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pactools.grid_search import GridSearchCVProgressBar
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin

class SparseToArray(BaseEstimator,TransformerMixin):

    def transform(self, X):
        # what other output you want
        return X.toarray()

    def fit(self, X, y=None, **fit_params):
        return self

# %%
df = pd.read_pickle(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")

# decrease dataset size
fraction_of_data = 0.2
df = df.sample(frac=fraction_of_data, replace=False, random_state=42)

# text
X = df["text"]
# label
df["outlier_label"] = df["target"]
y = df["outlier_label"]
y = (y * -1).clip(lower=0)
y[y == 0] = -1


pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('vectorize', 'passthrough'),
    ('reduce_dim', umap.UMAP(random_state=42)),
    ('classify', LocalOutlierFactor(novelty=True, contamination=0.183))
])


MIN_DF = [50]
MIX_RATIO = [0.1, 0.18]
N_COMPONENTS = [20, 50, 100]
METRICS = ['hellinger']
param_grid = [
    {
        'vectorize': [TfidfVectorizer(stop_words='english'), CountVectorizer(stop_words='english')],
        'vectorize__min_df': MIN_DF,
        'reduce_dim__set_op_mix_ratio': MIX_RATIO,
        'reduce_dim__n_components': N_COMPONENTS,
        'reduce_dim__metric': METRICS
    }
]

grid = GridSearchCV(pipe, scoring='f1', param_grid=param_grid, cv=3, verbose=10, n_jobs=-1)
grid.fit(X, y)

out_df = pd.DataFrame.from_dict(grid.cv_results_)
out_df = out_df.sort_values(by=['rank_test_score'])
out_df.to_csv("/home/philipp/projects/dad4td/reports/eval_4.tsv", sep="\t")

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from gensim.sklearn_api import D2VTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from eval_utils import TaggedDocsTransformer, GLOSHTransformer, HDBSCANPredictor
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, homogeneity_score, v_measure_score, completeness_score


pipes = {
    "base_pipe": Pipeline([
        ('vectorize', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('classify', 'passthrough')
    ]),
    "d2v_pipe": Pipeline([
        ('tagged_docs', 'passthrough'),
        ('vectorize', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('classify', 'passthrough')
    ]),
    "hdbscan": Pipeline([
        ('vectorize', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('cluster', 'passthrough')
    ]),
    "hdbscan_d2v": Pipeline([
        ('tagged_docs', 'passthrough'),
        ('vectorize', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('cluster', 'passthrough')
    ])
}



def get_pgrid(key, con=0.1, seed=42):
    pgrids = {
        "umap_params": {
            'vectorize': [TfidfVectorizer(stop_words='english')],
            'vectorize__min_df': [10, 25, 75],
            'reduce_dim': [UMAP(random_state=seed)],
            'reduce_dim__n_components': [2, 15, 50, 200],
            'reduce_dim__set_op_mix_ratio': [0.0, 0.05, 0.1],
            'reduce_dim__metric': ['euclidean'],
            'classify': [LocalOutlierFactor(novelty=True, contamination=con)],
            'classify__metric': ['euclidean']

        },

        "no_red_params": {
            'vectorize': [TfidfVectorizer(stop_words='english')],
            'vectorize__min_df': [25],
            'reduce_dim': ["passthrough"],
            'classify': [GLOSHTransformer()],
            # 'classify__metric': ['euclidean']
        },

        "d2v_no_red": {
            'tagged_docs': [TaggedDocsTransformer()],
            'tagged_docs__lower': [False],
            'vectorize': [D2VTransformer(seed=seed)],
            'vectorize__size': [100],
            'vectorize__window': [5],
            'reduce_dim': ["passthrough"],
            'classify': [GLOSHTransformer()]
        },

        "d2v_red": {
            'tagged_docs': [TaggedDocsTransformer()],
            'vectorize': [D2VTransformer(seed=seed)],
            'vectorize__min_count': [25],
            'vectorize__size': [50, 100, 300],
            'vectorize__window': [5, 10],
            'reduce_dim': [UMAP(random_state=seed)],
            'reduce_dim__n_components': [5, 50, 200],
            'reduce_dim__set_op_mix_ratio': [0.0],
            'reduce_dim__metric': ['euclidean'],
            'classify': [LocalOutlierFactor(novelty=True, contamination=con)],
            'classify__metric': ['euclidean']
        },

        "d2v_no_red_iso": {
            'tagged_docs': [TaggedDocsTransformer()],
            'vectorize': [D2VTransformer(seed=seed)],
            'vectorize__size': [50, 100, 300],
            'vectorize__window': [5, 10],
            'reduce_dim': ["passthrough"],
            'classify': [IsolationForest(contamination=con, random_state=seed)]
        },

        "d2v_red_iso": {
            'tagged_docs': [TaggedDocsTransformer()],
            'vectorize': [D2VTransformer(seed=seed)],
            'vectorize__min_count': [25],
            'vectorize__size': [50, 100, 300],
            'vectorize__window': [5, 10],
            'reduce_dim': [UMAP(random_state=seed)],
            'reduce_dim__n_components': [5, 50, 200],
            'reduce_dim__set_op_mix_ratio': [0.0],
            'reduce_dim__metric': ['euclidean'],
            'classify': [IsolationForest(contamination=con, random_state=seed)]
        },
        "hdbscan": {
            'vectorize': [TfidfVectorizer(stop_words='english')],
            'vectorize__min_df': [25],
            'reduce_dim': [UMAP(random_state=seed)],
            'reduce_dim__n_components': [10, 15],
            'reduce_dim__set_op_mix_ratio': [0.0, 0.5, 1.0],
            'reduce_dim__metric': ['euclidean'],
            'cluster': [HDBSCANPredictor()],
            'cluster__metric': ['euclidean'],
            'cluster__min_cluster_size': [10, 15],
            'cluster__no_noise': [True]

        },
        "hdbscan_d2v": {
            'tagged_docs': [TaggedDocsTransformer()],
            'vectorize': [D2VTransformer(seed=seed)],
            'vectorize__min_count': [25],
            'vectorize__size': [100],
            'vectorize__window': [5],
            'reduce_dim': [UMAP(random_state=seed)],
            'reduce_dim__n_components': [15],
            'reduce_dim__set_op_mix_ratio': [0.0, 0.5, 1.0],
            'reduce_dim__metric': ['euclidean'],
            'cluster': [HDBSCANPredictor()],
            'cluster__metric': ['euclidean'],
            'cluster__min_cluster_size': [10, 15],
            'cluster__no_noise': [True]

        },
        "hdbscan_d2v_no_red": {
            'tagged_docs': [TaggedDocsTransformer()],
            'vectorize': [D2VTransformer(seed=seed)],
            'vectorize__min_count': [25],
            'vectorize__size': [100],
            'vectorize__window': [5],
            'cluster': [HDBSCANPredictor()],
            'cluster__metric': ['euclidean'],
            'cluster__min_cluster_size': [10, 15],
            'cluster__no_noise': [True]

        }
    }
    return pgrids[key]


pipe_and_grid = {
    "basic": (pipes["base_pipe"],
              [get_pgrid("umap_params"), get_pgrid("no_red_params")]),
    "no_red_basic": (pipes["base_pipe"],
                     [get_pgrid("no_red_params")]),
    "umap_basic": (pipes["base_pipe"],
                   [get_pgrid("umap_params")]),
    "d2v_no_red": (pipes["d2v_pipe"],
                   [get_pgrid("d2v_no_red")]),
    "d2v_red": (pipes["d2v_pipe"],
                [get_pgrid("d2v_red")]),
    "d2v_no_red_iso": (pipes["d2v_pipe"],
                       [get_pgrid("d2v_no_red_iso")]),
    "d2v_red_iso": (pipes["d2v_pipe"],
                    [get_pgrid("d2v_red_iso")]),
    "hdbscan": (pipes["hdbscan"],
                [get_pgrid("hdbscan")]),
    "hdbscan_d2v": (pipes["hdbscan_d2v"],
                    [get_pgrid("hdbscan_d2v")]),
    "hdbscan_d2v_no_red": (pipes["hdbscan_d2v"],
                    [get_pgrid("hdbscan_d2v_no_red")])

}


eval_runs = {
    "eval_01": [pipe_and_grid["no_red_basic"], pipe_and_grid["umap_basic"], pipe_and_grid["d2v_no_red"]],
    "d2v_dims": [pipe_and_grid["d2v_red_iso"], pipe_and_grid["d2v_no_red_iso"], pipe_and_grid["d2v_no_red"], pipe_and_grid["d2v_red"]],
    "eval_dv_red": [pipe_and_grid["d2v_red"]],
    "d2v_no_red": [pipe_and_grid["d2v_no_red"]],
    "no_red_basic": [pipe_and_grid["no_red_basic"]],
    "hdbscan": [pipe_and_grid["hdbscan"]],
    "hdbscan_d2v": [pipe_and_grid["hdbscan_d2v_no_red"], pipe_and_grid["hdbscan_d2v"]]
}

# scoring metrics in reverse order
scorers = {
    "base": {"refit_metric": "f1_macro",
             "scoring_funcs": {
                 "in_f1": make_scorer(f1_score, pos_label=1),
                 "out_rec": make_scorer(recall_score, pos_label=-1),
                 "out_prec": make_scorer(precision_score, pos_label=-1),
                 "out_f1": make_scorer(f1_score, pos_label=-1),
                 "f1_micro": "f1_micro",
                 "f1_macro": "f1_macro"
             }
             },
    "cluster": {"refit_metric": "homogeneity_score",
                "scoring_funcs": {
                    "completeness_score": "completeness_score",
                    "v_measure_score": "v_measure_score",
                    "homogeneity_score": "homogeneity_score"
                }
                }
}

from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from gensim.sklearn_api import D2VTransformer
from sklearn.neighbors import LocalOutlierFactor
from eval_utils import TaggedDocsTransformer
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score


pipes = {
    "pipe": Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('vectorize', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('classify', 'passthrough')
    ]),
    "d2v_pipe": Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('tagged_docs', 'passthrough'),
        ('vectorize', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('classify', 'passthrough')
    ])
}


def get_pgrid(key, con=0.1, seed=42):
    pgrids = {
        "umap_params": {
            'vectorize': [TfidfVectorizer(stop_words='english')],
            'vectorize__min_df': [10, 25],
            'reduce_dim': [UMAP(random_state=seed)],
            'reduce_dim__n_components': [2, 15, 50, 200],
            'reduce_dim__set_op_mix_ratio': [0.0, 0.05, 0.1],
            'reduce_dim__metric': ['euclidean'],
            'classify': [LocalOutlierFactor(novelty=True, contamination=con)],
            'classify__metric': ['euclidean']

        },

        "no_red_params": {
            'vectorize': [TfidfVectorizer(stop_words='english')],
            'vectorize__min_df': [10, 25],
            'reduce_dim': ["passthrough"],
            'classify': [LocalOutlierFactor(novelty=True, contamination=con)],
            'classify__metric': ['euclidean']
        },

        "d2v_no_red": {
            'tagged_docs': [TaggedDocsTransformer()],
            'vectorize': [D2VTransformer(seed=seed)],
            'vectorize__min_count': [10, 25],
            'reduce_dim': ["passthrough"],
            'classify': [LocalOutlierFactor(novelty=True, contamination=con)],
            'classify__metric': ['euclidean']
        }
    }
    return pgrids[key]


pipe_and_grid = {
    "basic": (pipes["d2v_pipe"],
              [get_pgrid("d2v_no_red"), get_pgrid("d2v_no_red")]),
    "d2v": (pipes["d2v_pipe"],
            [get_pgrid("d2v_no_red")])
}


eval_runs = {
    "eval_01": [pipe_and_grid["d2v"], pipe_and_grid["basic"]]
}

# scoring metrics in reverse order
scorers = {
            "in_f1": make_scorer(f1_score, pos_label=1),
            "out_rec": make_scorer(recall_score, pos_label=-1),
            "out_prec": make_scorer(precision_score, pos_label=-1),
            "out_f1": make_scorer(f1_score, pos_label=-1),
            "f1_micro": "f1_micro",
            "f1_macro": "f1_macro"
            }
from timeit import default_timer as timer
from collections import defaultdict
from eval_utils import next_path
from tqdm import tqdm
import pandas as pd
import plotly.express as px
from umap import UMAP
from eval_cluster_config import TestData, Doc2VecModel, PyodDetector
from pyod.models.ocsvm import OCSVM

tqdm.pandas(desc="progess: ")


def add_scores(scores, list_of_param_dicts):
    for param_dict in list_of_param_dicts:
        for key, value in zip(param_dict, param_dict.values()):
            scores[key] = value
    return scores

def get_result(row):
    if row["outlier_label"] == 1 and row["predicted"] == 1:
        return "inlier - true positive"
    if row["outlier_label"] == -1 and row["predicted"] == -1:
        return "outlier - true negative"
    if row["outlier_label"] == -1 and row["predicted"] == 1:
        return "false negative (outlier predicted as inlier)"
    if row["outlier_label"] == 1 and row["predicted"] == -1:
        return "false positive (inlier predicted as outlier)"
    return "-1"

def load_coords_to_df(df, coords_2d):
    if "X" not in df and "Y" not in df:
        df["X"] = coords_2d[:, 0]
        df["Y"] = coords_2d[:, 1]

    return df


def prepare_text(df, col, line_chars=75):
    textlen = line_chars * 30
    df["htext"] = df["text"].str.replace(r'\\n', '<br>', regex=True)
    df["htext"] = df["htext"].map(lambda x: "<br>".join(
        x[i:i+line_chars] for i in range(0, len(x), line_chars)))
    df["htext"] = df["htext"].str[0: textlen]

    df["char_count"] = df["text"].apply(len)

    return df


def create_show_graph(df, col, coords_2d=None, color="title", line_chars=75, kwargs={}):
    df = load_coords_to_df(df, coords_2d)
    df = prepare_text(df, col)

    default_kwargs = {'x': 'X', 'y': 'Y', 'color': color, 'hover_data': ["title", "htext", "char_count"],
                      'color_discrete_sequence': px.colors.qualitative.Dark24, 'color_discrete_map': {"-1": "rgb(255, 255, 255)"}}
    default_kwargs.update(kwargs)

    print("Create graph ...")
    fig = px.scatter(df, **default_kwargs)
    return fig


# parameters
set_op_mix_ratio = 1.0
test_data = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[], contamination=[], seed=[])
doc2vec_model = Doc2VecModel("doc2vecapnews", "wiki_EN", 1.0,
                             100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")
dim_reducer = UMAP(metric="cosine", set_op_mix_ratio=set_op_mix_ratio,
               n_components=100, random_state=42)

# get test data

test_data.load_data().remove_short_texts()
test_data.sample_data(fraction=1.0, contamination=0.1, seed=1)

# vectorize

docvecs = doc2vec_model.vectorize(test_data.df["text"])

# dim reduce
dim_reduced_vecs = dim_reducer.fit_transform(docvecs)

# outlier prediction
scores = defaultdict(list)
outlier_predictor = PyodDetector(OCSVM, "OCSVM")
scores, preds = outlier_predictor.predict(dim_reduced_vecs, scores, test_data.df["outlier_label"], 0.1, "OCSVM")

# create visualization
vecs_2d = UMAP(metric="cosine", set_op_mix_ratio=set_op_mix_ratio,
               n_components=2, random_state=42).fit_transform(docvecs)

df = test_data.df
df["predicted"] = preds
df["result"] = df.apply(lambda row: get_result(row), axis=1)
fig = create_show_graph(
    test_data.df, "text", coords_2d=vecs_2d, color="result")

print(scores)
fig.show()

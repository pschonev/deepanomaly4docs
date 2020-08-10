# %%
import pandas as pd
from umap import UMAP
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm
import plotly.express as px

tqdm.pandas(desc="my bar!")

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

    default_kwargs = {'x':'X', 'y':'Y', 'color':color, 'hover_data':["title", "htext", "char_count"],
                     'color_discrete_sequence':px.colors.qualitative.Dark24, 'color_discrete_map':{"-1": "rgb(255, 255, 255)"}}
    default_kwargs.update(kwargs)

    print("Create graph ...")
    fig = px.scatter(df, **default_kwargs)
    return fig


# parameters
data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"
model_path = "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin"

set_op_mix_ratio = 0 # value between 0 and 1

# prepare data
print("Get data...")
df = pd.read_pickle(data_path)
X = df["text"]

# text lowered and split into list of tokens
print("Pre-process data...")
X = X.progress_apply(lambda x: simple_preprocess(x))

print("TaggedDocuments being prepared...")
tagged_docs = [TaggedDocument(doc, [i]) for i, doc in X.items()]

print("Train Doc2Vec model...")
model = Doc2Vec.load(model_path)

print("Infer doc vectors...")
docvecs = X.progress_apply(lambda x: model.infer_vector(x))
docvecs = list(docvecs)

print("dim reduction 2D ...")
vecs_2d = UMAP(metric="cosine", set_op_mix_ratio=set_op_mix_ratio,
               n_components=2, random_state=42).fit_transform(docvecs)

fig = create_show_graph(df, "text", coords_2d=vecs_2d, color="result")
fig.show()
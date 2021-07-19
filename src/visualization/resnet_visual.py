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

    default_kwargs = {'x':'X', 'y':'Y', 'color':color, 'hover_data':["htext"],
                     'color_discrete_sequence':px.colors.qualitative.Dark24, 'color_discrete_map':{"-1": "rgb(255, 255, 255)"}}
    default_kwargs.update(kwargs)

    print("Create graph ...")
    fig = px.scatter(df, **default_kwargs)
    return fig


# parameters
#data_path = "/media/philipp/Fotos/rvl-cdip/rvl_cdip_embs.pkl"
data_path = "/media/philipp/Fotos/rvl-cdip/rvl_cdip_vgg.pkl"
model_path = "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin"


# %% prepare data
n_per_targ = 500

print("Get data...")
df = pd.read_pickle(data_path)
print("Sample data...")
df = df.groupby('target', group_keys=False).apply(
        lambda df: df.sample(n=min(df.shape[0], n_per_targ), random_state=42))
df.vecs = df.vecs.map(lambda x: x.flatten())
df["text"] = df.target.astype(str)
df.target = df.target.astype(str)
df
# %%
set_op_mix_ratio = 0 # value between 0 and 1

#docvecs = df.resnet18_emb.to_list()
docvecs = df.vecs.to_list()

print("dim reduction 2D ...")
vecs_2d = UMAP(metric="cosine", set_op_mix_ratio=set_op_mix_ratio,
               n_components=2, random_state=42).fit_transform(docvecs)

fig = create_show_graph(df, "text", coords_2d=vecs_2d, color="target")
fig.show()
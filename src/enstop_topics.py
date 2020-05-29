# %%
import umap
from enstop import EnsembleTopics
from features.feat_utils import load_data, get_tf_idf, get_new_path, loadcreate
from visualization.visualize import create_show_graph


# add logging
# better colors?
# size = length of doc?

# %%
text_col = 'text'
data_path = "/home/philipp/projects/dad4td/data/external/20_newsgroup/20_newsgroup.csv"
emb_path = get_new_path(file_path=data_path, new_folder="data/processed",
                        suffix="_enstop_umap", file_ext="npy")
topics_path = get_new_path(file_path=data_path, new_folder="data/processed",
                           suffix="_enstop_topics", file_ext="npy")

df = load_data(data_path, dropna=True)
tfidf_word_doc_matrix = get_tf_idf(df, text_col)

# %%
topic_embeddings = loadcreate(tfidf_word_doc_matrix, topics_path, EnsembleTopics(
    n_components=20, parallelism='none'))


# %%
embeddings_2d = loadcreate(
    topic_embeddings, emb_path, umap.UMAP(metric='hellinger'))
# %%
create_show_graph(df, text_col, coords_2d=embeddings_2d)

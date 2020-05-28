# %%
from features.feat_utils import loadcreate_umap_emb, load_data, get_tf_idf, get_new_path
from visualization.visualize import create_show_graph
from sklearn.neighbors import LocalOutlierFactor

text_col = 'text'
data_path = "/home/philipp/projects/dad4td/data/external/20_newsgroup/20_newsgroup.csv"
emb_path = get_new_path(
    file_path=data_path, new_folder="data/processed",   suffix="_umap_out", file_ext="npy")
out_text_path = get_new_path(
    file_path=data_path, new_folder="reports", suffix="_outliers_raw", file_ext="txt")

# %%
df = load_data(data_path, dropna=True)
tfidf_word_doc_matrix = get_tf_idf(df, text_col)
tfidf_2d_emb = loadcreate_umap_emb(
    tfidf_word_doc_matrix, emb_path, umap_kwargs={'set_op_mix_ratio': 0.25})

# %%
outlier_scores = LocalOutlierFactor().fit_predict(tfidf_2d_emb)

# %%
df["outlier"] = outlier_scores

sep = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
outlier_texts = df["text"][df["outlier"] == -1]

with open(out_text_path, "w") as out_text_file:
    out_text_file.write(sep.join(outlier_texts))

# %%
df["outlier"] = df["outlier"] + 1
df["outlier"] = df["outlier"].astype(str)

create_show_graph(df, text_col, coords_2d=tfidf_2d_emb, kwargs={
                  "color": "outlier", "symbol": "title"})

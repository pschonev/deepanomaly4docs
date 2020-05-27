# %%
from features.feat_utils import loadcreate_umap_emb, load_data, get_tf_idf, get_feat_path
from visualization.visualize import create_show_graph
from sklearn.neighbors import LocalOutlierFactor

data_path = "/home/philipp/projects/dad4td/data/external/20_newsgroup/20_newsgroup.csv"
text_col = 'text'
emb_path = get_feat_path(data_path, suffix="_umap_out")

# %%
df = load_data(data_path, dropna=True)
tfidf_word_doc_matrix = get_tf_idf(df, text_col)
tfidf_2d_emb = loadcreate_umap_emb(tfidf_word_doc_matrix, emb_path, umap_kwargs={'set_op_mix_ratio': 0.25})

# %% 
outlier_scores = LocalOutlierFactor().fit_predict(tfidf_2d_emb)
outlying_digits = tfidf_2d_emb[outlier_scores == -1]
print(outlying_digits.shape)
print(outlier_scores.shape)

df["outlier"] = outlier_scores
df["outlier"] = df["outlier"] + 1
df["outlier"] = df["outlier"].astype(str)

# %%
create_show_graph(df, text_col, coords_2d=tfidf_2d_emb, kwargs={"color": "outlier", "symbol":"title"})


# %%

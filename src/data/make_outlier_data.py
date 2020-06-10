# %%
import pandas as pd

df_20 = pd.read_csv("/home/philipp/projects/dad4td/data/external/20_newsgroup/20_newsgroup.csv")
df_imdb = pd.read_csv("/home/philipp/projects/dad4td/data/external/imdb/IMDB Dataset.csv")

# %%

df_20 = df_20[["text", "target", "title"]]
df_20

# %%
df_imdb= df_imdb.rename(columns={'review':'text'})
df_imdb = df_imdb[["text"]]
df_imdb["title"] = "imdb"
df_imdb["target"] = -1
df_imdb

# %%

df_comb = pd.concat([df_20, df_imdb], ignore_index = True)
df_comb

# %%
df_comb.info()

# %%
df_comb = df_comb.dropna()
df_comb["text"] = df_comb["text"].astype(str)
df_comb["title"] = df_comb["title"].astype(str)


df_comb["outlier_label"] = (df_comb["target"] * -1).clip(lower=0)
df_comb["outlier_label"][df_comb["outlier_label"] == 0] = -1
df_comb
# %%

# imdb has 50.000 entries, 20 newsgroup only 11.000
# 20 newsgroup is divied by categories, imdb does not have genre information (in this csv at least)

df_comb.to_pickle("/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")

# %%

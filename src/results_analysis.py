# %%
import pandas as pd 

df = pd.read_csv("/home/philipp/projects/dad4td/reports/one_class/one_out_0002.tsv", sep="\t", index_col=0)
df.fillna(0).groupby(["cclass", "weakly"]).mean()

# %%
import numpy as np
df.where(df.weakly.isna()).groupby(["cclass"]).mean()

# %%
df_sup = pd.read_csv("/home/philipp/projects/dad4td/reports/supervised/one_new_outlier_weakly0001.tsv", sep="\t", index_col=0)
df_sup.where(df_sup.weakly_supervised==0).dropna().groupby(["test_outliers"]).mean()

#%%
df = pd.read_csv("/home/philipp/projects/dad4td/reports/one_class/one_out_img_0001.tsv", sep="\t", index_col=0)
df.fillna(0).groupby(["cclass", "weakly"]).mean()
df.where(df.weakly.isna()).groupby(["cclass"]).mean()

# %%
import pandas as pd

df = pd.read_csv(
    "/home/philipp/projects/dad4td/reports/clustering/0002_cluster_eval_new.tsv", sep="\t")

df_mean = df.groupby("hash").mean()
df_mean = df_mean.sort_values("f1_macro")
df_mean

score_cols = ['completeness',
              'f1_macro',
              'homogeneity',
              'in_f1',
              'out_f1',
              'out_prec',
              'out_rec',
              'v_measure']
data_cols = ['Unnamed: 0',
             'contamination',
             'data_frac',
             'seed']
param_cols = [x for x in list(df.columns) if x not in score_cols]
param_cols = [x for x in param_cols if x not in data_cols]
cols = score_cols + param_cols

df_mean = df.groupby("hash")[score_cols].mean()
df_mean = df_mean.reset_index()
df_mean = df_mean.sort_values(by="out_f1", ascending=False).reset_index(drop=True)
df_mean.head(20)
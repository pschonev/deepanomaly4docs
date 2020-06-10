# %%
import pandas as pd
import numpy as np

data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"
data_frac = 0.1
outlier_balance = 0.1
seed = 42
# %%
df = pd.read_pickle(data_path)
for i in range(1,10):
    df[f"test_{i}"] = 0.1
df.info()
# %%
X_n = int(df.shape[0] * data_frac)
y_n = int(X_n * outlier_balance)

df = df.iloc[np.random.RandomState(seed=42).permutation(len(df))]
df["outlier_label"] = (df["target"] * -1).clip(lower=0)
df = df[df["outlier_label"] == 1].head(X_n).append(
    df[df["outlier_label"] == 0].head(y_n))

X = df["text"]
y = df["outlier_label"]
# %% decrease dataset size
fraction_of_data = 0.1
df = df.sample(frac=fraction_of_data, replace=False, random_state=42)

# text
X = df["text"]
# label
df["outlier_label"] = df["target"]
y = df["outlier_label"]
y = (y * -1).clip(lower=0)
y[y == 0] = -1

# %%

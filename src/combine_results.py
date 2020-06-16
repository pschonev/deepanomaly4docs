import pandas as pd
from pathlib import Path

result_dir = Path("/home/philipp/projects/dad4td/reports/density_estimation")
result_file = "/home/philipp/projects/dad4td/reports/density_estimations.tsv"

files = result_dir.glob("*tsv")
df = pd.concat((pd.read_csv(f, sep="\t") for f in files))
df = df.sort_values(by=['mean_test_f1_macro'], ascending=False).reset_index(drop=True)

if True:
    start_col = ("std", "split","rank","Unnamed", "seed")
    drop_cols = [c for c in df.columns if c.startswith(start_col)]
    df = df.drop(drop_cols, axis=1)

df = df[df.columns[::-1]]
df = df.drop_duplicates().reset_index(drop=True)
df.to_csv(result_file, sep="\t")

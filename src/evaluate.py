import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from eval_utils import sample_data, save_data
from eval_config import eval_runs
from sklearn.metrics import make_scorer, f1_score

# parameters
data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"

data_params = dict(data_frac=0.1,
                   contamination=0.1,
                   seed=42)

# prepare data
# ! creation of datasets should probably be handled seperately all within a class/function
df = pd.read_pickle(data_path)
# class for imdb_20news that lets me choose 20 news categories?
df = sample_data(df, **data_params)

X = df["text"]
y = df["outlier_label"]

tot_str = ""
for pipe_and_grid in eval_runs["test"]:
    # prepare pipeline
    pipe, param_grid = pipe_and_grid[0], pipe_and_grid[1]

    # grid search
    scorer = {"f1_macro": "f1_macro",
              "f1_micro": "f1_micro",
              "in_f1": make_scorer(f1_score, pos_label=-1),
              "out_f1": make_scorer(f1_score, pos_label=1)}

    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=data_params["seed"])

    grid = GridSearchCV(pipe, scoring=scorer, param_grid=param_grid,
                        cv=cv, verbose=10, n_jobs=-1, refit='f1_macro')
    grid.fit(X, y)

    # save results and params to file
    param_str = "\n".join(str(x) for x in [data_path, data_params, param_grid]) # !!! when combining results, remove most rows for better overview
    save_data(results_df=grid.cv_results_,
              data_params=data_params, param_str=param_str)

    tot_str = param_str if tot_str == "" else tot_str + \
        f"\n\n---------------------------\n\n{param_str}"

print(tot_str)

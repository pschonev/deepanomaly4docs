import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from eval_utils import save_data
from eval_config import eval_runs, scorers
from data.datasets import get_out_data


data_params = dict(dataset_name="imdb_20news",
                   data_frac=0.1, contamination=0.1, seed=42)
X, y = get_out_data(**data_params)

tot_str = ""
for pipe_and_grid in eval_runs["test"]:
    # prepare pipeline
    pipe, param_grid = pipe_and_grid[0], pipe_and_grid[1]

    # grid search
    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=data_params["seed"])

    grid = GridSearchCV(pipe, scoring=scorers, param_grid=param_grid,
                        cv=cv, verbose=10, n_jobs=-1, refit='f1_macro')
    grid.fit(X, y)

    # save results and params to file
    # !!! when combining results, remove most rows for better overview
    param_str = "\n".join(str(x) for x in [data_params, grid])
    save_data(results_df=grid.cv_results_,
              data_params=data_params, param_str=param_str)

    tot_str = param_str if tot_str == "" else tot_str + \
        f"\n\n---------------------------\n\n{param_str}"

print(tot_str)

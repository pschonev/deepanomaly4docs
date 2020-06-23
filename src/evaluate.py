import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from eval_utils import save_data
from eval_config import eval_runs, scorers
from data.datasets import get_out_data


eval_name = "hdbscan_d2v"
scoring_name = "cluster"
n_splits = 2
n_jobs = 1

eval_res_folder = "/home/philipp/projects/dad4td/reports/clustering/"
eval_res_pattern = "%04d_cluster_eval.tsv"

data_params = dict(dataset_name="imdb_20news",
                   data_frac=0.2, contamination=0.1, subset="", seed=42)
X, y = get_out_data(**data_params)

scoring = scorers[scoring_name]["scoring_funcs"]
refit_metric = scorers[scoring_name]["refit_metric"]

tot_str = ""
for pipe_and_grid in eval_runs[eval_name]:
    print(pipe_and_grid)
    # prepare pipeline
    pipe, param_grid = pipe_and_grid[0], pipe_and_grid[1]

    # grid search
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=data_params["seed"])

    grid = GridSearchCV(pipe, scoring=scoring, param_grid=param_grid,
                        cv=cv, verbose=10, n_jobs=n_jobs, refit=refit_metric)
    grid.fit(X, y)

    # save results and params to file
    # !!! when combining results, remove most rows for better overview
    param_str = "\n".join(str(x) for x in [data_params, grid])
    save_data(results_df=grid.cv_results_,
              data_params=data_params, param_str=param_str, sort_by=f"rank_test_{refit_metric}", res_folder=eval_res_folder, res_pattern=eval_res_pattern)

    tot_str = param_str if tot_str == "" else tot_str + \
        f"\n\n---------------------------\n\n{param_str}"

print(tot_str)

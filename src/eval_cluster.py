from timeit import default_timer as timer
from collections import defaultdict
from eval_utils import next_path
from functools import reduce
from operator import mul
from tqdm import tqdm
import pandas as pd
from eval_cluster_config import eval_runs

tqdm.pandas(desc="progess: ")

def prod(iterable):
    return reduce(mul, iterable, 1)


# parameters
eval_run = eval_runs["new_test"]

# !!! MOVE THIS INTO THE CLASS !!!
result_path = next_path(eval_run.res_folder + "%04d_" + eval_run.name + ".tsv")
print(f"Saving results to {result_path}")

# initialize variables
scores = defaultdict(list)
result_df = pd.DataFrame()


# !!!! FIX THIS !!!!!
data_params = eval_run.test_data.__dict__
test_params = eval_run.test_settings.__dict__

total_i = prod(len(x) for x in test_params.values())
total_ik = len(eval_run.models) * total_i
total_ikj = prod(len(x) for x in data_params.values()) * total_ik

for i, test_data in enumerate(eval_run.test_datasets):
    # load the test data from provided path and remove texts that are too short
    test_data.load_data().remove_short_texts()

    for j, data_params_ in enumerate(test_data.cartesian_params()):
        # sample with given parameters
        test_data.sample_data(test_data.df, **data_params_)

        for k, model in enumerate(eval_run.models):

            # get document vectors from model
            docvecs = model.vectorize(test_data.df["text"])

            for l, dim_reduction in enumerate(eval_run.dim_reductions):
                for m, dim_reduction_params_ in enumerate(test_data.cartesian_params()):
                    start = timer()

                    # dimension reduction on the vectors, if dim_reduction object is not None
                    if dim_reduction:
                        dim_reduced_vecs = dim_reduction.reduce_dims()
                    else:
                        dim_reduced_vecs = docvecs

                    for n, outlier_predictor in enumerate(eval_run.outlier_detectors):
                        for o, outlier_predictor_params in enumerate(outlier_predictor.cartesian_params()):
                            
                            # !!!!!!! DO THIS !!!
                            # displaying parameters
                            data_param_str = ", ".join(
                                [f"{key}: {value}" for key, value in zip(data_params, data_params_)])
                            model_param_str = ", ".join(
                                [f"{key}: {value}" for key, value in zip(model.__dict__, model.__dict__.values())])
                            test_param_str = ", ".join(
                                [f"{key}: {value}" for key, value in zip(test_params, test_params_)])
                            print(
                                f"run {j*total_ik + k*total_i + i+1} out of {total_ikj} --- {model_param_str}  {data_param_str} | {test_param_str}")

                            # !! LOOK AT THIS AT LEAST !!
                            # adding param values to results dict
                            for key, value in zip(test_params, test_params_):
                                scores[key] = value
                            for key, value in zip(model.__dict__, model.__dict__.values()):
                                scores[key] = value
                            for key, value in zip(data_params, data_params_):
                                scores[key] = value

                            # time
                            end = timer()
                            scores["time"] = end-start

                            # save results and print output
                            result_df = result_df.append(scores, ignore_index=True)
                            results_df.to_csv(result_path, sep="\t")

print(result_df)
print(f"Saved results to {result_path}")

from timeit import default_timer as timer
from collections import defaultdict
from eval_utils import next_path
from tqdm import tqdm
import pandas as pd
from eval_cluster_config import eval_runs

tqdm.pandas(desc="progess: ")

def add_scores(scores, list_of_param_dicts):
    for param_dict in list_of_param_dicts:
        for key, value in zip(param_dict, param_dict.values()):
            scores[key] = value
    return scores

# parameters
eval_run = eval_runs["new_test_complete"]

# initialize variables
eval_run.init_result_path()
eval_run.init_iter_counter()

scores = defaultdict(list)
result_df = pd.DataFrame()

for i, test_data in enumerate(eval_run.test_datasets):
    # load the test data from provided path and remove texts that are too short
    test_data.load_data().remove_short_texts()

    for j, data_params_ in enumerate(test_data.cartesian_params()):
        # sample with given parameters
        test_data.sample_data(**data_params_)

        for k, model in enumerate(eval_run.models):

            # get document vectors from model
            docvecs = model.vectorize(test_data.df["text"])

            for l, dim_reduction in enumerate(eval_run.dim_reductions):
                for m, dim_reduction_params_ in enumerate(dim_reduction.cartesian_params()):
                    start = timer()

                    print(
                        f"\n{str(data_params_)} \n \
                        {str(model)} \n \
                        {str(dim_reduction_params_)}\n\n\
                        ---------------------------------------------------\n")

                    # dimension reduction on the vectors, if dim_reduction object is not None
                    dim_reduced_vecs = dim_reduction.reduce_dims(
                        docvecs, **dim_reduction_params_)

                    for n, outlier_predictor in enumerate(eval_run.outlier_detectors):
                        for o, outlier_predictor_params in enumerate(outlier_predictor.cartesian_params()):

                            print(f"run {eval_run.current_iter} out of {eval_run.total_iter}\n\
                                {str(outlier_predictor_params)}\n\
                                    -------------------")

                            scores = outlier_predictor.predict(dim_reduced_vecs, scores, test_data.df["outlier_label"], data_params_[
                                                               "contamination"], **outlier_predictor_params)
                            eval_run.current_iter += 1

                    add_scores(scores, [data_params_, model.__dict__, dim_reduction_params_, outlier_predictor_params])

                    # time
                    end = timer()
                    scores["time"] = end-start

                    # save results and print output
                    result_df = result_df.append(
                        scores, ignore_index=True)
                    result_df.to_csv(eval_run.res_path, sep="\t")

print(result_df)
print(f"Saved results to {eval_run.res_path}")

from timeit import default_timer as timer
from collections import defaultdict
from eval_utils import next_path
from tqdm import tqdm
import pandas as pd
from evaluation_config import eval_runs

tqdm.pandas(desc="progess: ")

def add_scores(scores, list_of_param_dicts):
    for param_dict in list_of_param_dicts:
        for key, value in zip(param_dict, param_dict.values()):
            scores[key] = value
    return scores

# parameters
eval_run = eval_runs["invis_params"]

# initialize variables
eval_run.init_result_path()
eval_run.init_iter_counter()

result_df = pd.DataFrame()

for test_data in eval_run.test_datasets:
    # load the test data from provided path and remove texts that are too short
    test_data.load_data().remove_short_texts()

    for data_params_ in test_data.cartesian_params():
        # sample with given parameters
        df = test_data.sample_data(**data_params_)

        for model in eval_run.models:

            # get document vectors from model
            docvecs = model.vectorize(df["text"])

            for dim_reduction in eval_run.dim_reductions:
                for dim_reduction_params_ in dim_reduction.cartesian_params():
                    print(
                        f"\n{str(data_params_)} \n\n \
                        {str(model)} \n\n \
                        {str(dim_reduction)} \n {str(dim_reduction_params_)}\n\n\
                        ---------------------------------------------------\n")

                    # dimension reduction on the vectors
                    print("Reducing dims...")
                    start_dim_reduce = timer()
                    dim_reduced_vecs = dim_reduction.reduce_dims(
                        docvecs, **dim_reduction_params_)
                    end_dim_reduce = timer()
                    time_dim_reduce = end_dim_reduce - start_dim_reduce
                    print(f"Reduction done - {time_dim_reduce} ms")

                    for outlier_predictor in eval_run.outlier_detectors:
                        for outlier_predictor_params in outlier_predictor.cartesian_params():
                            scores = defaultdict(list)
                            start = timer()
                            print(f"\nrun {eval_run.current_iter} out of {eval_run.total_iter}\n\
                                {str(outlier_predictor_params)}\n\
                                    -------------------")
                            
                            scores,_ = outlier_predictor.predict(dim_reduced_vecs, df["outlier_label"], scores, data_params_[
                                                               "contamination"], **outlier_predictor_params)
                            eval_run.current_iter += 1

                            add_scores(scores, [data_params_, model.__dict__, dim_reduction.__dict__, dim_reduction_params_, outlier_predictor_params])
                            #scores["dim_reducer"] = dim_reduction
                            scores["dim_reduce_time"] = time_dim_reduce

                            # time
                            end = timer()
                            scores["time"] = end-start

                            # save results and print output
                            result_df = result_df.append(
                                scores, ignore_index=True)
                            result_df.to_csv(eval_run.res_path, sep="\t")

print(result_df)
print(f"Saved results to {eval_run.res_path}")

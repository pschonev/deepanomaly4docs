from src.supervised.supervised import PassThroughTransformer, SupEvalRun
from src.utils import get_scores
from typing import TypeVar
import pandas as pd

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')


def eval_sup(eval_run: SupEvalRun, result_df: PandasDataFrame, c:dict) -> None:
    data = eval_run.data
    data.prep_vecs(data.data_prep.inliers, c)
    eval_run.transformer.create_model(data, c)

    best_acc = 0
    print("training...")
    # vector transformation train step loop
    for epochnumber in range(c.epoch_num):
        eval_run.transformer.train_step(data, c)

        # outlier prediction
        if epochnumber % c.epoch_report == 0 or epochnumber == c.epoch_num-1:
            
            # prepare train and test data
            train, test, y_tr, test_labels = data.get_train_test_vecs()
            
            # transform vectors
            train = eval_run.transformer.transform(train)
            test = eval_run.transformer.transform(test)

            # train outlier predictor with train data and predict outlier scores on test data
            decision_scores = eval_run.od_predictor.fit_predict(c, train, test, y_tr, data, pass_through=isinstance(eval_run.transformer, PassThroughTransformer))

            # add scores
            scores = get_scores(test_labels, decision_scores,
                                outlabel=0, threshold=c.threshold)
            print(f"\n\nTest scores:\n{pd.DataFrame([scores], index=[0])}")
            if epochnumber == 0:
                best_scores=scores
            if scores["pr_auc"] > best_acc and epochnumber != 0:
                best_acc = scores["pr_auc"]
                best_scores = scores
                print(f"best_acc updated to: {best_acc}")

    # remove neural networks from memory
    eval_run.transformer.cleanup()

    # save results
    result_df = result_df.append(dict(cclass=list(
        data.database.df_test.target.unique()), **best_scores, **c.__dict__), ignore_index=True)
    result_df.to_csv(eval_run.res_path, sep="\t")
    result_df.to_pickle(eval_run.res_path[:-3]+"pkl")

    return result_df


def grid_search(eval_run: SupEvalRun) -> None:
    data = eval_run.data

    eval_run.init_result_path()
    data.prep_data()
    data.database.vectorize_data(data.txt_vecs, data.img_vecs, data.mode)

    result_df = pd.DataFrame()
    param_combs = list(eval_run.parameters.param_combinations(eval_mode=eval_run.eval_mode, outliers=data.data_prep.outliers, transformer=eval_run.transformer))
    for i, c in enumerate(param_combs):
        chosen_class_log = f"Chosen class: {c.chosen_class}" if c.chosen_class else ""
        print(f"\n--------\nRun {i+1} out of {len(param_combs)}. {chosen_class_log}\nParams: {c}\n")
        result_df = eval_sup(eval_run, result_df, c)

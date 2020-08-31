# %%
import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from gensim.sklearn_api import D2VTransformer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize
from itertools import product
from functools import reduce
from operator import mul
from collections import defaultdict, OrderedDict
from timeit import default_timer as timer
from eval_utils import next_path
from eval_cluster_config import eval_runs
from tqdm import tqdm

tqdm.pandas(desc="progess: ")


def prod(iterable):
    return reduce(mul, iterable, 1)


def sample_data(df, data_frac, contamination, seed):
    X_n = int(df.shape[0] * data_frac)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    df = df.reset_index(drop=True)
    return df


def get_result(row):
    if row["outlier_label"] == 1 and row["predicted"] == 1:
        return "inlier - true positive"
    if row["outlier_label"] == -1 and row["predicted"] == -1:
        return "outlier - true negative"
    if row["outlier_label"] == -1 and row["predicted"] == 1:
        return "false negative (outlier predicted as inlier)"
    if row["outlier_label"] == 1 and row["predicted"] == -1:
        return "false positive (inlier predicted as outlier)"
    return "-1"


def doc2vec_vectors(X, model_path):
    # text lowered and split into list of tokens
    print("Pre-process data...")
    X = X.progress_apply(lambda x: simple_preprocess(x))

    # load model
    print("Load Doc2Vec model...")
    model = Doc2Vec.load(model_path)

    # print("TaggedDocuments being prepared...")
    # tagged_docs = [TaggedDocument(doc, [i]) for i, doc in X.items()]

    # model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    # model.build_vocab(all_docs_tagged)

    print("Infer doc vectors...")
    docvecs = X.progress_apply(lambda x: model.infer_vector(x))
    return list(docvecs)


def bert_doc_embeddings(X, bert_model):
    # init embedding model
    print("Load BERT model ...")
    model = TransformerDocumentEmbeddings(bert_model, fine_tune=False)

    # convert to Sentence objects
    print("Convert to Sentence objects ...")
    X = X.str.lower()
    sentences = X.progress_apply(lambda x: Sentence(x))

    # get vectors from BERT
    print("Get BERT embeddings ...")
    docvecs = sentences.progress_apply(lambda x: model.embed(x))
    docvecs = sentences.progress_apply(lambda x: x.embedding.cpu().numpy())
    return list(docvecs)


# parameters
eval_run = eval_runs["doc2vecwikiimdb20news013030"]
result_path = next_path(eval_run.res_folder + "%04d_" + eval_run.name + ".tsv")
print(f"Saving results to {result_path}")

# load data and remove empty texts
print("Get data...")
df_all = pd.read_pickle(eval_run.test_data.test_data_path[0])
n_before = df_all.shape[0]
df_all = df_all[df_all['text'].map(len) > eval_run.min_doc_length]
print(
    f"Removed {n_before - df_all.shape[0]} rows with doc length below {eval_run.min_doc_length}.")

# initialize variables
scores = defaultdict(list)
result_df = pd.DataFrame()

data_params = eval_run.test_data.__dict__
models = eval_run.model
test_params = eval_run.test_settings.__dict__

total_i = prod(len(x) for x in test_params.values())
total_ik = len(eval_run.model) * total_i
total_ikj = prod(len(x) for x in data_params.values()) * total_ik

for j, data_params_ in enumerate(product(*data_params.values())):
    _, _, data_frac, contamination, seed = data_params_

    # sample
    df = sample_data(df_all, data_frac=data_frac,
                     contamination=contamination, seed=seed)
    X = df["text"]

    for k, model in enumerate(models):

        # load model
        if model.model_type.lower() == "doc2vec":
            docvecs = doc2vec_vectors(X, model.model_path)
        elif model.model_type.lower() == "transformer":
            docvecs = bert_doc_embeddings(X, bert_model=model.model_name)

        for i, test_params_ in enumerate(product(*test_params.values())):
            start = timer()
            n_comp, mix_ratio, umap_metric, min_cluster_size, allow_noise = test_params_

            # displaying parameters
            data_param_str = ", ".join(
                [f"{key}: {value}" for key, value in zip(data_params, data_params_)])
            model_param_str = ", ".join(
                [f"{key}: {value}" for key, value in zip(model.__dict__, model.__dict__.values())])
            test_param_str = ", ".join(
                [f"{key}: {value}" for key, value in zip(test_params, test_params_)])
            print(
                f"run {j*total_ik + k*total_i + i+1} out of {total_ikj} --- {model_param_str}  {data_param_str} | {test_param_str}")

            # pipeline
            if n_comp != -1:
                dim_reduced_vecs = UMAP(metric=umap_metric, set_op_mix_ratio=mix_ratio,
                                        n_components=n_comp, random_state=42).fit_transform(docvecs)
            else:
                dim_reduced_vecs = docvecs

            # LocalOutlierFactor
            print("Get LocalOutlierFactor...")
            df["predicted_LOF"] = LocalOutlierFactor(
                novelty=False, metric=umap_metric, contamination=contamination).fit_predict(dim_reduced_vecs)

            # clustering
            print("Clustering ...")
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                                prediction_data=True, metric="euclidean").fit(dim_reduced_vecs)
            print("Get prediction data ...")
            clusterer.generate_prediction_data()

            # scoring
            print("Get scores ...")

            # GLOSH
            threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
            df["predicted"] = np.where(
                clusterer.outlier_scores_ > threshold, -1, 1)

            # cluster scoring
            outlier_labels = df["outlier_label"]
            outlier_pred = df["predicted"]
            outlier_pred_LOF = df["predicted_LOF"]
            try:
                cluster_pred = clusterer.labels_ if allow_noise else np.argmax(
                    all_points_membership_vectors(clusterer)[:, 1:], axis=1)
            except IndexError:
                print(
                    "Got IndexError and will not enforce cluster membership (allow noise) ...")
                cluster_pred = clusterer.labels_

            # adding param values to results dict
            for key, value in zip(test_params, test_params_):
                scores[key] = value
            for key, value in zip(model.__dict__, model.__dict__.values()):
                scores[key] = value
            for key, value in zip(data_params, data_params_):
                scores[key] = value

            # scores
            scores["homogeneity"] = homogeneity_score(
                outlier_labels, cluster_pred)
            scores["completeness"] = completeness_score(
                outlier_labels, cluster_pred)
            scores["v_measure"] = v_measure_score(outlier_labels, cluster_pred)
            scores["f1_macro"] = f1_score(
                outlier_labels, outlier_pred, average='macro')
            scores["in_f1"] = f1_score(
                outlier_labels, outlier_pred, pos_label=1)
            scores["out_f1"] = f1_score(
                outlier_labels, outlier_pred, pos_label=-1)
            scores["out_rec"] = recall_score(
                outlier_labels, outlier_pred, pos_label=-1)
            scores["out_prec"] = precision_score(
                outlier_labels, outlier_pred, pos_label=-1)
            scores["f1_macro_LOF"] = f1_score(
                outlier_labels, outlier_pred_LOF, average='macro')
            scores["in_f1_LOF"] = f1_score(
                outlier_labels, outlier_pred_LOF, pos_label=1)
            scores["out_f1_LOF"] = f1_score(
                outlier_labels, outlier_pred_LOF, pos_label=-1)
            scores["out_rec_LOF"] = recall_score(
                outlier_labels, outlier_pred_LOF, pos_label=-1)
            scores["out_prec_LOF"] = precision_score(
                outlier_labels, outlier_pred_LOF, pos_label=-1)

            # time
            end = timer()
            scores["time"] = end-start

            # unique hash for params (without data params)
            scores["hash"] = "|".join([str(x) for x in test_params_])

            scores["cluster_n"] = len(np.unique(clusterer.labels_))

            # save results and print output
            result_df = result_df.append(scores, ignore_index=True)
            results_df = result_df.sort_values(
                by=["homogeneity"]).reset_index(drop=True)
            results_df.to_csv(result_path, sep="\t")

            print(f"Homogeneity - {homogeneity_score(outlier_labels, cluster_pred)*100:.1f}  \
                    f1_macro - {f1_score(outlier_labels, outlier_pred, average='macro')*100:.1f}  \
                    out_f1 - {f1_score(outlier_labels, outlier_pred, pos_label=-1)*100:.1f}   \
                    out_f1_LOF {f1_score(outlier_labels, outlier_pred_LOF, pos_label=-1)*100:.1f} \
                    cluster_n - {len(np.unique(clusterer.labels_))} \
                    time - {end-start: .1f} \n\n - ------------------\n")

print(result_df)
print(f"Saved results to {result_path}")

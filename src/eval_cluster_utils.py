
from sklearn.neighbors import LocalOutlierFactor
from hdbscan import HDBSCAN, all_points_membership_vectors
import numpy as np


def remove_short_texts(df, min_len):
    n_before = df.shape[0]
    df = df[df['text'].map(len) > min_len]
    print(
        f"Removed {n_before - df.shape[0]} rows with doc length below {min_len}.")
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

    # infer vectors from model
    print("Infer doc vectors...")
    docvecs = X.progress_apply(lambda x: model.infer_vector(x))
    return list(docvecs)


def bert_doc_embeddings(X, transformer_model):
    # init embedding model
    print(f"Load {transformer_model} model ...")
    model = TransformerDocumentEmbeddings(transformer_model, fine_tune=False)

    # convert to Sentence objects
    print("Convert to Sentence objects ...")
    X = X.str.lower()
    sentences = X.progress_apply(lambda x: Sentence(x))

    # get vectors from BERT
    print("Get BERT embeddings ...")
    docvecs = sentences.progress_apply(lambda x: model.embed(x))
    docvecs = sentences.progress_apply(lambda x: x.embedding.cpu().numpy())
    return list(docvecs)

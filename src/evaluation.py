from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import List, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score
from flair.embeddings import TransformerDocumentEmbeddings, WordEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.neighbors import LocalOutlierFactor
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from ivis import Ivis
from sklearn.decomposition import PCA as PCAR
from sklearn.manifold import TSNE
from flair.data import Sentence
from umap import UMAP


def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = '%03d-results.tsv':

    001-results.tsv
    001-results.tsv
    """
    i = 1

    # First do an exponential search
    while Path(path_pattern % i).exists():
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if Path(path_pattern % c).exists() else (a, c)

    return path_pattern % b


def product_dict(**kwargs):
    return [dict(zip(kwargs.keys(), x)) for x in product(*kwargs.values())]


def get_scores(scores, outlier_labels, outlier_pred):
    scores[f"f1_macro"] = f1_score(
        outlier_labels, outlier_pred, average='macro')
    scores[f"in_f1"] = f1_score(
        outlier_labels, outlier_pred, pos_label=1)
    scores[f"in_rec"] = recall_score(
        outlier_labels, outlier_pred, pos_label=1)
    scores[f"in_prec"] = precision_score(
        outlier_labels, outlier_pred, pos_label=1)
    scores[f"out_f1"] = f1_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_rec"] = recall_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_prec"] = precision_score(
        outlier_labels, outlier_pred, pos_label=-1)
    return scores


def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = np.quantile(sr, [pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return ((np.abs(sr - median)) >= iqr/2), median, iqr


def sample_data(df, fraction, contamination, seed):
    X_n = int(df[df.outlier_label==1].shape[0] * fraction)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    df = df.reset_index(drop=True)
    return df


@dataclass
class TestData:
    path: str
    name: str
    fraction: List[float]
    contamination: List[float]
    seed: List[int]
    min_len: int = 5

    df: pd.DataFrame = None

    def load_data(self):
        print(f"Loading data from {self.path} to DataFrame...")
        self.df = pd.read_pickle(self.path)
        return self

    def remove_short_texts(self):
        n_before = self.df.shape[0]
        self.df = self.df[self.df['text'].map(len) > self.min_len]
        print(
            f"Removed {n_before - self.df.shape[0]} rows with doc length below {self.min_len}.")
        return self

    def cartesian_params(self):
        return list(product_dict(fraction=self.fraction, contamination=self.contamination, seed=self.seed))

    def sample_data(self, fraction, contamination, seed):
        return sample_data(self.df, fraction, contamination, seed)


# model for conversion from text to vectors
@dataclass
class EmbeddingModel(ABC):
    # doc2vec, or huggingface transformer specifier (e.g. bert-uncased)
    model_name: str
    model_train_data: str

    @abstractmethod
    def vectorize(self, X):
        pass


@dataclass
class Doc2VecModel(EmbeddingModel):
    doc2vec_data_frac: float
    doc2vec_epochs: int
    doc2vec_min_count: int
    model_path: str
    model_type: str = "doc2vec"

    def __post_init__(self):
        # load model
        print("Load Doc2Vec model...")
        self.model = Doc2Vec.load(self.model_path)


    def vectorize(self, X):
        # text lowered and split into list of tokens
        print("Pre-process data...")
        X = X.progress_apply(lambda x: simple_preprocess(x))

        # infer vectors from model
        print("Infer doc vectors...")
        docvecs = X.progress_apply(lambda x: self.model.infer_vector(x))
        return list(docvecs)


@dataclass
class WordEmbeddingPooling(EmbeddingModel):
    model_type: str = "wordembeddingpool"

    @staticmethod
    def embed(x, model, dim):
        try:
            model.embed(x)
            return x.embedding.detach().cpu().numpy()
        except RuntimeError:
            return np.zeros(dim)

    def vectorize(self, X):
        # init embedding model
        print(f"Load {self.model_name} model ...")
        w_emb = WordEmbeddings(self.model_name)
        model = DocumentPoolEmbeddings([w_emb], fine_tune_mode='nonlinear')

        # convert to Sentence objects
        print("Convert to Sentence objects ...")
        X = X.str.lower()
        sentences = X.progress_apply(lambda x: Sentence(x))

        # get vectors from BERT
        print(f"Get {self.model_name} embeddings ...")
        docvecs = sentences.progress_apply(lambda x: self.embed(
            x, model, model.embedding_flex.out_features))
        docvecs = np.vstack(docvecs)
        return list(docvecs)


@dataclass
class RNNEmbedding(EmbeddingModel):
    model_path: str
    model_type: str = "grnn"

    @staticmethod
    def embed(x, model, dim):
        try:
            model.embed(x)
            return x.get_embedding().detach().cpu().numpy()
        except RuntimeError:
            return np.zeros(dim)

    def vectorize(self, X):
        # init embedding model
        print(f"Load {self.model_name} model ...")
        classifier = TextClassifier.load(self.model_path)
        model = classifier.document_embeddings

        # convert to Sentence objects
        print("Convert to Sentence objects ...")
        X = X.str.lower()
        sentences = X.progress_apply(lambda x: Sentence(x))

        # get vectors from BERT
        print(f"Get {self.model_name} embeddings ...")
        docvecs = sentences.progress_apply(lambda x: self.embed(
            x, model, classifier.document_embeddings.embedding_length))
        docvecs = np.vstack(docvecs)
        return list(docvecs)


@dataclass
class TransformerModel(EmbeddingModel):
    model_size_params: int
    model_type: str = "transformer"

    def vectorize(self, X):
        # init embedding model
        print(f"Load {self.model_name} model ...")
        model = TransformerDocumentEmbeddings(self.model_name, fine_tune=False)

        # convert to Sentence objects
        print("Convert to Sentence objects ...")
        X = X.str.lower()
        sentences = X.progress_apply(lambda x: Sentence(x))

        # get vectors from BERT
        print(f"Get {self.model_name} embeddings ...")
        docvecs = sentences.progress_apply(lambda x: model.embed(x))
        docvecs = sentences.progress_apply(lambda x: x.embedding.cpu().numpy())
        return list(docvecs)


class DimensionReducer(ABC):
    name: str

    @abstractmethod
    def reduce_dims(self, docvecs):
        pass


class NoReduction(DimensionReducer):
    def cartesian_params(self):
        return [dict()]

    def reduce_dims(self, docvecs):
        return docvecs


@dataclass
class SklearnReducer(DimensionReducer):
    dim_reducer: Any
    as_numpy: bool
    kwargs: dict = field(default_factory=dict)

    def cartesian_params(self):
        return product_dict(**self.kwargs)

    def reduce_dims(self, docvecs, **kwargs):
        if self.as_numpy:
            docvecs = np.array(docvecs)
        return self.dim_reducer(**kwargs).fit_transform(docvecs)


@dataclass
class OutlierDetector(ABC):
    @abstractmethod
    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, **kwargs):
        pass


@dataclass
class PyodDetector(OutlierDetector):
    pyod_model: Any
    outlier_detector: str
    kwargs: dict = field(default_factory=dict)

    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, **kwargs):
        print(f"Outlier detection using pyod's {self.pyod_model}")
        od = self.pyod_model(**kwargs)
        od.fit(dim_reduced_vecs)

        out_pred = od.labels_
        out_pred[out_pred == 1] = -1
        out_pred[out_pred == 0] = 1

        scores = get_scores(scores, outlier_labels, out_pred)
        scores.update(**kwargs)
        out_f1 = scores["out_f1"]
        print(f"{kwargs}\nOut_f1: {out_f1}\n\n")
        return scores, out_pred

    def cartesian_params(self):
        return product_dict(**self.kwargs)


@dataclass
class DimRedOutlierDetector(OutlierDetector):
    dem_red_outlier_model: Any
    outlier_detector: str
    as_numpy: bool
    kwargs: dict = field(default_factory=dict)

    @staticmethod
    def reject_outliers(sr, iq_range=0.5):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = np.quantile(sr, [pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        return ((np.abs(sr - median)) >= iqr/2)

    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, **kwargs):
        od = self.dem_red_outlier_model(**kwargs)
        if self.as_numpy:
            dim_reduced_vecs = np.array(dim_reduced_vecs)
        preds = od.fit_transform(dim_reduced_vecs)
        preds = preds.astype(float)

        preds = self.reject_outliers(preds, iq_range=1.0-contamination)
        preds = [-1 if x else 1 for x in preds]

        scores = get_scores(scores, outlier_labels, preds)
        scores.update(**kwargs)
        out_f1 = scores["out_f1"]
        print(f"{kwargs}\nOut_f1: {out_f1}\n\n")
        return scores, preds

    def cartesian_params(self):
        return product_dict(**self.kwargs)


@dataclass
class HDBSCAN_GLOSH(OutlierDetector):
    min_cluster_size: List[int]
    allow_noise: List[bool]
    outlier_detector: str = "HDBSCAN_GLOSH"

    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, min_cluster_size, allow_noise):
        print("Clustering ...")
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            prediction_data=True, metric="euclidean").fit(dim_reduced_vecs)
        print("Get prediction data ...")
        clusterer.generate_prediction_data()

        try:
            cluster_pred = clusterer.labels_ if allow_noise else np.argmax(
                all_points_membership_vectors(clusterer)[:, 1:], axis=1)
        except IndexError:
            print(
                "Got IndexError and will not enforce cluster membership (allow noise) ...")
            print(all_points_membership_vectors(clusterer))
            cluster_pred = clusterer.labels_

        # scoring
        print("Get scores ...")

        # GLOSH
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
        outlier_pred = np.where(
            clusterer.outlier_scores_ > threshold, -1, 1)

        scores["cluster_n"] = len(np.unique(clusterer.labels_))
        scores["homogeneity"] = homogeneity_score(
            outlier_labels, cluster_pred)
        scores["completeness"] = completeness_score(
            outlier_labels, cluster_pred)
        scores["v_measure"] = v_measure_score(outlier_labels, cluster_pred)

        scores = get_scores(scores, outlier_labels, outlier_pred)

        print(f"Homogeneity - {homogeneity_score(outlier_labels, cluster_pred)*100:.1f}  \
                cluster_n - {len(np.unique(clusterer.labels_))}")

        return scores, clusterer.outlier_scores_

    def cartesian_params(self):
        return product_dict(min_cluster_size=self.min_cluster_size, allow_noise=self.allow_noise)


@dataclass
class EvalRun:
    name: str
    models: List[EmbeddingModel]
    test_datasets: List[TestData]
    dim_reductions: List[DimensionReducer]
    outlier_detectors: List[OutlierDetector]
    res_folder: str = "/home/philipp/projects/dad4td/reports/eval_runs/"
    res_path: str = ""
    total_iter: int = 0
    current_iter: int = 1

    def init_iter_counter(self):
        model_perm = len(self.models)
        test_data_perm = sum(len(x.cartesian_params())
                             for x in self.test_datasets)
        dim_red_perm = sum(len(x.cartesian_params())
                           for x in self.dim_reductions)
        out_detect_perm = sum(len(x.cartesian_params())
                              for x in self.outlier_detectors)
        self.total_iter = model_perm*test_data_perm*dim_red_perm*out_detect_perm

        print(f"Evaluating {self.total_iter} parameter permutations.")
        return self

    def init_result_path(self):
        self.res_path = next_path(
            self.res_folder + "%04d_" + self.name + ".tsv")
        print(f"Saving results to {self.res_path}")
        return self

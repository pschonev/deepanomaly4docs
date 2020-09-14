from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import List, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score
from flair.embeddings import TransformerDocumentEmbeddings
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
    scores[f"out_f1"] = f1_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_rec"] = recall_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_prec"] = precision_score(
        outlier_labels, outlier_pred, pos_label=-1)
    return scores


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
        df = self.df
        X_n = int(df.shape[0] * fraction)
        y_n = int(X_n * contamination)

        df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
        df = df[df["outlier_label"] == 1].head(X_n).append(
            df[df["outlier_label"] == -1].head(y_n))
        df = df.reset_index(drop=True)
        return df


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

    def vectorize(self, X):
        # text lowered and split into list of tokens
        print("Pre-process data...")
        X = X.progress_apply(lambda x: simple_preprocess(x))

        # load model
        print("Load Doc2Vec model...")
        model = Doc2Vec.load(self.model_path)

        # infer vectors from model
        print("Infer doc vectors...")
        docvecs = X.progress_apply(lambda x: model.infer_vector(x))
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
        print("Get BERT embeddings ...")
        docvecs = sentences.progress_apply(lambda x: model.embed(x))
        docvecs = sentences.progress_apply(lambda x: x.embedding.cpu().numpy())
        return list(docvecs)


class DimensionReducer(ABC):
    @abstractmethod
    def reduce_dims(self, docvecs):
        pass


dataclass


class NoReduction(DimensionReducer):
    dim_reducer: str

    def cartesian_params(self):
        return [dict()]

    def reduce_dims(self, docvecs):
        return docvecs


@dataclass
class UMAPModel(DimensionReducer):
    set_op_mix_ratio: List[float]
    metric: List[str]
    n_components: List[int]
    dim_reducer: str = "umap"

    def cartesian_params(self):
        return product_dict(n_components=self.n_components, set_op_mix_ratio=self.set_op_mix_ratio,
                            metric=self.metric)

    def reduce_dims(self, docvecs, metric, set_op_mix_ratio, n_components):
        return UMAP(metric=metric, set_op_mix_ratio=set_op_mix_ratio,
                    n_components=n_components, random_state=42).fit_transform(docvecs)


@dataclass
class PCAReducer(DimensionReducer):
    n_components: List[int]
    dim_reducer: str = "PCA"

    def cartesian_params(self):
        return product_dict(n_components=self.n_components)

    def reduce_dims(self, docvecs, n_components):
        return PCAR(n_components=n_components).fit_transform(docvecs)


@dataclass
class TSNEReducer(DimensionReducer):
    n_components: List[int]
    perplexity: List[int]
    dim_reducer: str = "TSNE"

    def cartesian_params(self):
        return product_dict(n_components=self.n_components, perplexity=self.perplexity)

    def reduce_dims(self, docvecs, n_components, perplexity):
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(docvecs)


@dataclass
class OutlierDetector(ABC):
    @abstractmethod
    def predict(self, dim_reduced_vecs, scores, outlier_labels, contamination):
        pass


@dataclass
class PyodDetector(OutlierDetector):
    pyod_model: Any
    outlier_detector: str
    kwargs: dict = None

    def predict(self, dim_reduced_vecs, scores, outlier_labels, contamination, outlier_detector):
        print(f"Outlier detection using pyod's {self.pyod_model}")
        od = self.pyod_model(**self.kwargs)
        od.fit(dim_reduced_vecs)
        out_pred = od.labels_
        out_pred[out_pred == 1] = -1
        out_pred[out_pred == 0] = 1
        scores = get_scores(scores, outlier_labels, out_pred)
        scores.update(**self.kwargs)
        out_f1 = scores["out_f1"]

        print(f"out_f1 for {self.outlier_detector}: {out_f1*100:.1f}")
        return scores, out_pred

    def cartesian_params(self):
        return product_dict(outlier_detector=[self.outlier_detector])


@dataclass
class LOF_(OutlierDetector):
    metric: List[str]
    outlier_detector: str = "LOF_"

    def predict(self, dim_reduced_vecs, scores, outlier_labels, contamination, metric):
        print("Get LocalOutlierFactor...")
        outlier_pred = LocalOutlierFactor(
            novelty=False, metric=metric, contamination=contamination, n_jobs=-1).fit_predict(dim_reduced_vecs)

        scores = get_scores(scores, outlier_labels, outlier_pred)
        out_f1 = scores["out_f1"]

        print(f"out_f1 for {self.outlier_detector}: {out_f1*100:.1f}")

        return scores, outlier_pred

    def cartesian_params(self):
        return product_dict(metric=self.metric)


@dataclass
class HDBSCAN_GLOSH(OutlierDetector):
    min_cluster_size: List[int]
    allow_noise: List[bool]
    outlier_detector: str = "HDBSCAN_GLOSH"

    def predict(self, dim_reduced_vecs, scores, outlier_labels, contamination, min_cluster_size, allow_noise):
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
                f1_macro - {f1_score(outlier_labels, outlier_pred, average='macro')*100:.1f}  \
                out_f1 - {f1_score(outlier_labels, outlier_pred, pos_label=-1)*100:.1f}   \
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
    res_folder: str = "/home/philipp/projects/dad4td/reports/clustering/"
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


# vectorize model
doc2vecwikiall = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                              100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")
doc2vecapnews = Doc2VecModel("doc2vecapnews", "apnews", 1.0,
                             100, 1, "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin")
doc2vecwikiimdb20news011001 = Doc2VecModel("doc2vecwikiimdb20news011001", "wiki_EN", 0.1,
                                           10, 1, "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min1/doc2vec_wiki.bin")
longformer_large = TransformerModel(
    "allenai/longformer-large-4096", "long_documents", 435)

# test data
imdb_20news_3splits = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.15], contamination=[0.1], seed=[42, 43, 44])
imdb_20news_3splits_full = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[1.0], contamination=[0.1], seed=[42, 43, 44])
imdb_20news_3split_fracs = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.01, 0.05, 0.1], contamination=[0.1], seed=[42, 43, 44])
imdb_20news_3split_fracs_med = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.05, 0.1, 0.15], contamination=[0.1], seed=[42, 43, 44])


# dimension reduction
umap_all = UMAPModel([0.0, 0.15, 0.3, 0.6, 1.0], [
                     "cosine"], [3, 6, 15, 50, 100, 200])
umap_optim = UMAPModel([0.3, 0.6, 1.0], ["cosine"], [2, 3, 15, 50, 100, 200])
umap_big_dim = UMAPModel([0.6, 1.0], ["cosine"], [250, 300])
umap_small = UMAPModel([1.0], ["cosine"], [32, 50, 100])
umap_tiny = UMAPModel([1.0], ["cosine"], [4, 8, 16, 32])
umap_one_ae = UMAPModel([1.0], ["cosine"], [200])
umap_ae_ext = UMAPModel([0.5, 1.0], ["cosine"], [8, 16, 32, 64, 200])
pca_all = PCAReducer([3, 6, 15, 50, 100, 200])
pca_small = PCAReducer([8, 16])
tsne_all = TSNEReducer([3], [10, 30, 45])

# outlier detectors
lof_test = LOF_(["cosine"])

glosh_test = HDBSCAN_GLOSH([45], [False])

# eval run
pyod_test_umap = EvalRun("pyod_test_umap",
                         [doc2vecwikiall, doc2vecapnews,
                          doc2vecwikiimdb20news011001, longformer_large],
                         [imdb_20news_3splits],
                         [umap_all],
                         [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                             PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

pyod_test_no_red = EvalRun("pyod_test_no_red",
                           [doc2vecwikiall, doc2vecapnews,
                            doc2vecwikiimdb20news011001, longformer_large],
                           [imdb_20news_3splits],
                           [NoReduction()],
                           [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                            PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

dim_reducer_test = EvalRun("dim_reducer_test",
                           [doc2vecwikiall, doc2vecapnews,
                            doc2vecwikiimdb20news011001, longformer_large],
                           [imdb_20news_3splits],
                           [pca_all, tsne_all],
                           [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                            PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA"), glosh_test])

pyod_test_umap_all_data = EvalRun("pyod_test_umap_all_data",
                                  [doc2vecwikiall],
                                  [imdb_20news_3splits_full],
                                  [umap_optim],
                                  [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                                      PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

pyod_test_umap_all_data_no_red = EvalRun("pyod_test_umap_all_data_no_red",
                                         [longformer_large],
                                         [imdb_20news_3splits_full],
                                         [umap_optim, NoReduction()],
                                         [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                                             PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

pyod_umap_big_dim = EvalRun("pyod_umap_big_dim",
                            [doc2vecwikiall],
                            [imdb_20news_3splits],
                            [umap_big_dim],
                            [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                             PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

pyod_autoencoder_test = EvalRun("pyod_autoencoder_test",
                                [doc2vecwikiall, longformer_large],
                                [imdb_20news_3splits],
                                [umap_small, NoReduction()],
                                [PyodDetector(VAE(epochs=30, verbosity=1), "VAE_30"),
                                 PyodDetector(
                                     VAE(epochs=100, verbosity=1), "VAE_100"),
                                 PyodDetector(AutoEncoder(
                                     epochs=30, verbose=1), "AE_30"),
                                 PyodDetector(AutoEncoder(epochs=100, verbose=2), "AE_100")])

pyod_autoencer_refined = EvalRun("pyod_autoencer_refined",
                                 [doc2vecwikiall, doc2vecapnews],
                                 [imdb_20news_3split_fracs],
                                 [umap_small],
                                 [PyodDetector(AutoEncoder(hidden_neurons=[32, 16, 16, 32],
                                                           epochs=30, verbose=1), "AE_30_small"),
                                     PyodDetector(AutoEncoder(
                                         epochs=10, verbose=1), "AE_10"),
                                  PyodDetector(AutoEncoder(
                                      epochs=30, verbose=1), "AE_30"),
                                  PyodDetector(AutoEncoder(epochs=100, verbose=2), "AE_100")])

pyod_autoencer_refined_small = EvalRun("pyod_autoencer_refined_small",
                                       [doc2vecapnews, doc2vecwikiall,
                                           doc2vecwikiimdb20news011001],
                                       [imdb_20news_3split_fracs_med],
                                       [umap_tiny],
                                       [PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                                 epochs=10, verbose=1), "AE_10_tiny"),
                                        PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                                 epochs=30, verbose=1), "AE_30_tiny")
                                        ])

pyod_autoencer_refined_ext = EvalRun("pyod_autoencer_refined_ext",
                                     [doc2vecapnews],
                                     [imdb_20news_3split_fracs_med],
                                     [umap_ae_ext, pca_small],
                                     [PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 2, 2, 4, 8],
                                                               epochs=10, verbose=1), "AE_10_micro_3"),
                                      PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 4, 8],
                                                               epochs=5, verbose=1), "AE_5_micro"),
                                      PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 4, 8],
                                                               epochs=10, verbose=1), "AE_10_micro"),
                                      PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 4, 8],
                                                               epochs=30, verbose=1), "AE_30_micro"),
                                      PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 4, 8],
                                                               epochs=100, verbose=1), "AE_100_micro"),
                                      PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 4, 4, 8, 16],
                                                               epochs=10, verbose=1), "AE_10_tiny_3")
                                      ])

pyod_autoencer_full_data = EvalRun("pyod_autoencer_full_data",
                                   [doc2vecapnews],
                                   [imdb_20news_3splits_full],
                                   [NoReduction(), UMAPModel(
                                       [1.0], ["cosine"], [256, 299, 300])],
                                   [PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                             epochs=1, verbose=1), "AE_1_mono"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 4],
                                                             epochs=3, verbose=1), "AE_3_duo_as"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                             epochs=2, verbose=1), "AE_2_mono"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                             epochs=3, verbose=1), "AE_3_mono"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                             epochs=5, verbose=1), "AE_5_mono"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                             epochs=10, verbose=1), "AE_10_mono"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                             epochs=1, verbose=1), "AE_1_duo"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                             epochs=2, verbose=1), "AE_2_duo"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                             epochs=3, verbose=1), "AE_3_duo"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                             epochs=5, verbose=1), "AE_5_duo"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                             epochs=10, verbose=1), "AE_10_duo"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 4, 8],
                                                             epochs=1, verbose=1), "AE_1_micro"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 4, 8],
                                                             epochs=2, verbose=1), "AE_2_micro"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[8, 4, 4, 8],
                                                             epochs=3, verbose=1), "AE_3_micro"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                             epochs=1, verbose=1), "AE_1_tiny"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                             epochs=2, verbose=1), "AE_2_tiny"),
                                    PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                             epochs=3, verbose=1), "AE_3_tiny")
                                    ])

pyod_autoencer_full_data_no_red = EvalRun("pyod_autoencer_full_data_no_red",
                                          [doc2vecapnews],
                                          [imdb_20news_3splits_full],
                                          [NoReduction()],
                                          [PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                                    epochs=10, verbose=1), "AE_10_mono"),
                                           PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                                    epochs=50, verbose=1), "AE_50_mono"),
                                           PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                                    epochs=100, verbose=1), "AE_100_mono"),
                                           PyodDetector(AutoEncoder(hidden_neurons=[2, 1, 1, 2],
                                                                    epochs=300, verbose=1), "AE_300_mono"),
                                              PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                                       epochs=10, verbose=1), "AE_10_duo"),
                                              PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                                       epochs=50, verbose=1), "AE_50_duo"),
                                              PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                                       epochs=100, verbose=1), "AE_100_duo"),
                                              PyodDetector(AutoEncoder(hidden_neurons=[4, 2, 2, 4],
                                                                       epochs=300, verbose=1), "AE_300_duo"),
                                           PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                                    epochs=100, verbose=1), "AE_100_tiny")
                                           ])


pyod_autoencer_no_red_big = EvalRun("pyod_autoencer_no_red_big",
                                    [doc2vecapnews],
                                    [imdb_20news_3split_fracs_med],
                                    [NoReduction()],
                                    [PyodDetector(AutoEncoder, "AE", dict(hidden_neurons=[8, 2, 2, 8], epochs=10)),
                                     PyodDetector(AutoEncoder, "AE", dict(
                                         hidden_neurons=[8, 2, 2, 8], epochs=100)),
                                     PyodDetector(AutoEncoder, "AE", dict(
                                         hidden_neurons=[64, 32, 8, 8, 32, 64], epochs=10)),
                                     PyodDetector(AutoEncoder, "AE", dict(
                                         hidden_neurons=[64, 32, 8, 8, 32, 64], epochs=100)),
                                     PyodDetector(AutoEncoder, "AE", dict(
                                         hidden_neurons=[64, 32, 8, 2, 8, 32, 64], epochs=10)),
                                     PyodDetector(AutoEncoder, "AE", dict(hidden_neurons=[
                                                  64, 32, 8, 2, 8, 32, 64], epochs=100)),
                                     PyodDetector(AutoEncoder, "AE", dict(hidden_neurons=[
                                                  128, 64, 32, 8, 8, 32, 64, 128], epochs=10)),
                                     ])

pyod_autoencer_full_data_small = EvalRun("pyod_autoencer_full_data_small",
                                         [doc2vecapnews, doc2vecwikiall],
                                         [imdb_20news_3splits_full],
                                         [UMAPModel([1.0], ["cosine"], [
                                                    2, 4, 8, 32, 64, 256, 300])],
                                         [
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=1)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=3)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=5)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2], epochs=1)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2], epochs=3)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2], epochs=5)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2, 1, 2], epochs=1)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2, 1, 2], epochs=3)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2, 1, 2], epochs=5)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2, 1, 1, 2], epochs=1)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2, 1, 1, 2], epochs=3)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[2, 1, 1, 2], epochs=5))
                                         ])

pyod_autoencoder_mono = pyod_autoencer_full_data_small = EvalRun("pyod_autoencoder_mono",
                                         [doc2vecapnews, doc2vecwikiall],
                                         [imdb_20news_3splits_full],
                                         [UMAPModel([1.0], ["cosine"], [
                                                    1, 2, 4, 8, 32, 64, 256, 300]), NoReduction()],
                                         [
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=1)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=2)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=3)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=4)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=5)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=10)),
                                             PyodDetector(AutoEncoder, "AE", dict(
                                                 hidden_neurons=[1], epochs=100))
                                         ])

# dictionary containing all the settings
eval_runs = {
    "pyod_test_umap": pyod_test_umap,
    "dim_reducer_test": dim_reducer_test,
    "pyod_test_no_red": pyod_test_no_red,
    "pyod_test_umap_all_data": pyod_test_umap_all_data,
    "pyod_test_umap_all_data_no_red": pyod_test_umap_all_data_no_red,
    "pyod_autoencoder_test": pyod_autoencoder_test,
    "pyod_autoencer_refined": pyod_autoencer_refined,
    "pyod_autoencer_refined_small": pyod_autoencer_refined_small,
    "pyod_autoencer_refined_ext": pyod_autoencer_refined_ext,
    "pyod_autoencer_full_data": pyod_autoencer_full_data,
    "pyod_autoencer_full_data_no_red": pyod_autoencer_full_data_no_red,
    "pyod_autoencer_no_red_big": pyod_autoencer_no_red_big,
    "pyod_autoencer_full_data_small": pyod_autoencer_full_data_small,
    "pyod_autoencoder_mono": pyod_autoencoder_mono}

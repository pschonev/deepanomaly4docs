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

def get_scores(scores, outlier_labels, outlier_pred, desc, sep="_"):
    scores[f"f1_macro{sep}{desc}"] = f1_score(
        outlier_labels, outlier_pred, average='macro')
    scores[f"in_f1{sep}{desc}"] = f1_score(
        outlier_labels, outlier_pred, pos_label=1)
    scores[f"out_f1{sep}{desc}"] = f1_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_rec{sep}{desc}"] = recall_score(
        outlier_labels, outlier_pred, pos_label=-1)
    scores[f"out_prec{sep}{desc}"] = precision_score(
        outlier_labels, outlier_pred, pos_label=-1)
    return scores


@dataclass
class TestData:
    path: str
    name: str
    fraction: List[float]
    contamination: List[float]
    seed: List[int]
    min_len: int = 10

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
        self.df = df.reset_index(drop=True)
        return self

    


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

    def cartesian_params(self):
        return [dict()]

    def reduce_dims(self, docvecs):
        return docvecs


@dataclass
class UMAPModel(DimensionReducer):
    set_op_mix_ratio: List[float]
    metric: List[str]
    n_components: List[int]

    def cartesian_params(self):
        return product_dict(n_components=self.n_components, set_op_mix_ratio=self.set_op_mix_ratio,
                            metric=self.metric)

    def reduce_dims(self, docvecs, metric, set_op_mix_ratio, n_components):
        return UMAP(metric=metric, set_op_mix_ratio=set_op_mix_ratio,
                    n_components=n_components, random_state=42).fit_transform(docvecs)


class OutlierDetector(ABC):
    @abstractmethod
    def predict(self, dim_reduced_vecs, scores, outlier_labels, contamination):
        pass

class PyodDetector(OutlierDetector):
    pyod_model: Any
    def predict(self):
        pass

@dataclass
class LOF(OutlierDetector):
    metric: List[str]

    def predict(self, dim_reduced_vecs, scores, outlier_labels, contamination, metric):
        print("Get LocalOutlierFactor...")
        outlier_pred_LOF = LocalOutlierFactor(
            novelty=False, metric=metric, contamination=contamination, n_jobs=-1).fit_predict(dim_reduced_vecs)

        scores = get_scores(scores, outlier_labels, outlier_pred_LOF, "LOF") 
        out_f1 = scores["out_f1_LOF"]

        print(f"out_f1_LOF {out_f1*100:.1f}")

        return scores

    
    def cartesian_params(self):
        return product_dict(metric=self.metric)


@dataclass
class HDBSCAN_GLOSH(OutlierDetector):
    min_cluster_size: List[int]
    allow_noise: List[bool]


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

        scores = get_scores(scores, outlier_labels, outlier_pred, "", sep="")

        print(f"Homogeneity - {homogeneity_score(outlier_labels, cluster_pred)*100:.1f}  \
                f1_macro - {f1_score(outlier_labels, outlier_pred, average='macro')*100:.1f}  \
                out_f1 - {f1_score(outlier_labels, outlier_pred, pos_label=-1)*100:.1f}   \
                cluster_n - {len(np.unique(clusterer.labels_))}")

        return scores
    

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
    min_doc_length: int = 10
    total_iter: int = 0
    current_iter: int = 1

    def init_iter_counter(self):
        model_perm = len(self.models)
        test_data_perm = sum(len(x.cartesian_params()) for x in self.test_datasets)
        dim_red_perm = sum(len(x.cartesian_params()) for x in self.dim_reductions)
        out_detect_perm = sum(len(x.cartesian_params()) for x in self.outlier_detectors)
        self.total_iter = model_perm*test_data_perm*dim_red_perm*out_detect_perm

        print(f"Evaluating {self.total_iter} parameter permutations.")
        return self

    def init_result_path(self):
        self.res_path = next_path(self.res_folder + "%04d_" + self.name + ".tsv")
        print(f"Saving results to {self.res_path}")
        return self

# vectorize model
doc2vecwikiall = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                                  100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")

bert_base_uncased = TransformerModel("bert-base-uncased", "wiki_book", 110)

# test data
imdb_20news_3splits = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.15], contamination=[0.1], seed=[42, 43, 44])


# dimension reduction
umap_test = UMAPModel([0.15], ["cosine"], [3])

# outlier detectors
lof_test = LOF(["cosine"])

glosh_test = HDBSCAN_GLOSH([45], [False])

# eval run
new_test_complete= EvalRun("new_test_complete", [doc2vecwikiall, bert_base_uncased], [imdb_20news_3splits], [NoReduction(), umap_test], [glosh_test, lof_test])

# dictionary containing all the settings
eval_runs = {
    "new_test_complete": new_test_complete}

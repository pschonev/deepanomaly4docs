from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import List, Any
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


def product_dict(**kwargs):
    return (dict(zip(kwargs.keys(), x)) for x in product(kwargs.values()))

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
    min_len: int = 10
    fraction: List[float]
    contamination: List[float]
    seed: List[int]

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
        return product_dict(fraction=self.fraction, contamination=self.contamination, seed=self.seed)

    def sample_data(self, fraction, contamination, seed):
        df = self.df
        X_n = int(df.shape[0] * fraction)
        y_n = int(X_n * contamination)

        df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
        df = df[df["outlier_label"] == 1].head(X_n).append(
            df[df["outlier_label"] == -1].head(y_n))
        df = df.reset_index(drop=True)
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


@dataclass
class TestSettings:
    n_comps: List[int]
    mix_ratio: List[float]
    umap_metric: List[str]
    min_cluster_size: List[int]
    allow_noise: List[bool]


class DimensionReducer:
    pass


@dataclass
class UMAP(DimensionReducer):
    set_op_mix_ratio: List[float]
    metric: List[str]
    n_components: List[int]
    random_state: List[int] = [42]

    def cartesian_params(self):
        return product_dict(n_components=self.n_components, set_op_mix_ratio=self.set_op_mix_ratio,
                            metric=self.metric, random_state=self.random_state)

    def reduce_dims(self, docvecs, metric, set_op_mix_ratio, n_components, random_state):
        return UMAP(metric=metric, set_op_mix_ratio=set_op_mix_ratio,
                    n_components=n_components, random_state=42).fit_transform(docvecs)


class OutlierDetector(ABC):
    @abstractmethod
    def predict(self, X):
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

    def predict(self, dim_reduced_vecs, scores, outlier_labels, min_cluster_size, allow_noise):
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
    min_doc_length: int = 5


# model definitions

# doc2vec
# doc2vecwiki011030 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwiki011030", "/home/philipp/projects/dad4td/models/doc2vec_01_10/doc2vec_wiki.bin", "wikiEN", 0.1, 10, 30, None)
# doc2vecwiki013030 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwiki013030", "/home/philipp/projects/dad4td/models/doc2vec_01_30/doc2vec_wiki.bin", "wikiEN", 0.1, 30, 30, None)
# doc2vecwiki013001 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwiki013001", "/home/philipp/projects/dad4td/models/doc2vec_01_30_min1/doc2vec_wiki.bin", "wikiEN", 0.1, 30, 1, None)
# doc2vecwiki031030 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwiki031030", "/home/philipp/projects/dad4td/models/doc2vec_03_10/doc2vec_wiki.bin", "wikiEN", 0.3, 10, 30, None)
# doc2vecimdb20news101001 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecimdb20news101001", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_10_min1/doc2vec_wiki.bin", "imdb_20news", 1.0, 10, 1, None)
# doc2vecimdb20news1010001 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecimdb20news1010001", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_100_min1/doc2vec_wiki.bin", "imdb_20news", 1.0, 100, 1, None)
# doc2vecwikiimdb20news011001 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwikiimdb20news011001", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min1/doc2vec_wiki.bin", "wiki_imdb_20news", 0.1, 10, 1, None)
# doc2vecwikiimdb20news011030 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwikiimdb20news011030", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min30/doc2vec_wiki.bin", "wiki_imdb_20news", 0.1, 10, 30, None)
# doc2vecwikiimdb20news013030 = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwikiimdb20news013030", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min30/doc2vec_wiki.bin", "wiki_imdb_20news", 0.1, 30, 30, None)
# doc2vecapnews = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecapnews", "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin", "apnews", 1.0, 100, 1, None)
# doc2vecwikiall = EmbeddingModel(
#     "doc2vec", "doc2vec", "doc2vecwikiall", "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin", "wikiEN", 1.0, 100, 1, None)

doc2vecwikiall_new = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                                  100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")

# # transformer flair
# bert_base_uncased = EmbeddingModel(
#     "transformer", "bert-base-uncased", "bert-base-uncased", None, "EN_lower", None, None, None, 110)
# bert_large_uncased = EmbeddingModel(
#     "transformer", "bert-large-uncased", "bert-large-uncased", None, "EN_lower", None, None, None, 340)
# longformer_base = EmbeddingModel(
#     "transformer", "allenai/longformer-base-4096", "allenai/longformer-base-4096", None, "EN_lower", None, None, None, 149)
# longformer_large = EmbeddingModel(
#     "transformer", "allenai/longformer-large-4096", "allenai/longformer-large-4096", None, "EN_lower", None, None, None, 435)
# gpt2_large = EmbeddingModel(
#     "transformer", "gpt2-large", "gpt2-large", None, "EN_lower", None, None, None, 774)
# gpt2_medium = EmbeddingModel(
#     "transformer", "gpt2-medium", "gpt2-medium", None, "EN_lower", None, None, None, 345)
# roberta_large = EmbeddingModel(
#     "transformer", "roberta-large", "roberta-large", None, "EN_lower", None, None, None, 355)
# t5_large = EmbeddingModel(
#     "transformer", "t5-large", "t5-large", None, "EN_lower", None, None, None, 770)
# t5_base = EmbeddingModel(
#     "transformer", "t5-base", "t5-base", None, "EN_lower", None, None, None, 220)

bert_base_uncased_new = TransformerModel("bert-base-uncased", "wiki_book", 110)

# test data settings
imdb_20news_3splits = TestData(
    ["/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"], ["imdb_20news"], [0.15], [0.1], [42, 43, 44])


# evaluation settings
# standard_test = TestSettings([3, 6, 15, 45, 200], [0.0, 0.15, 0.3, 0.4, 1.0], [
#                              "cosine"], [15, 45, 90], [False])

# no_red_test = TestSettings([-1], [0.0, 0.15, 0.3, 0.4, 1.0], [
#     "cosine"], [15, 45, 90], [False])

# standard_and_no_red_test = TestSettings([-1, 3, 6, 15, 45, 200], [0.0, 0.15, 0.3, 0.4, 1.0], [
#     "cosine"], [15, 45, 90], [False])

# standard_big_clusters_test = TestSettings([3, 6, 15, 45, 200], [0.0, 0.15, 0.3, 0.4, 1.0], [
#     "cosine"], [120], [False])

# standard_bigger_clusters_test = TestSettings([3, 6, 15, 45, 200, -1], [0.15, 0.3, 0.4, 1.0], [
#     "euclidean", "cosine"], [250, 500], [False])

# small_transformer_test = TestSettings([-1], [0.15], ["cosine"], [250], [False])

small_test_new = TestSettings([3], [0.15], ["cosine"], [50], [False])

# evaluation run definition
# test_doc2vec = EvalRun("test_doc2vec", [
#                        doc2vecwiki011030, doc2vecwiki013030], imdb_20news_3splits, standard_test)

# full_doc2vec = EvalRun("full_doc2vec", [
#                        doc2vecwiki011030, doc2vecwiki013030, doc2vecwiki013001, doc2vecwiki031030, doc2vecimdb20news101001,
#                        doc2vecimdb20news1010001, doc2vecwikiimdb20news011001, doc2vecwikiimdb20news011030, doc2vecapnews, doc2vecwikiall
#                        ], imdb_20news_3splits, standard_test)

# doc2vecwikiimdb20news013030 = EvalRun("doc2vecwikiimdb20news013030", [
#     doc2vecwikiimdb20news013030], imdb_20news_3splits, standard_test)

# test_transformer = EvalRun("test_transformer", [
#                            bert_base_uncased, bert_large_uncased], imdb_20news_3splits, standard_test)

# transformer_no_red = EvalRun("transformer_no_red", [
#     bert_base_uncased, bert_large_uncased], imdb_20news_3splits, no_red_test)

# longformer_eval = EvalRun("longformer_eval", [
#     longformer_base, longformer_large], imdb_20news_3splits, standard_test)

# longformer_eval_big_clusters = EvalRun("longformer_eval_big_clusters", [
#     longformer_base, longformer_large], imdb_20news_3splits, standard_big_clusters_test)

# longformer_large_bigger_clusters = EvalRun("longformer_large_bigger_clusters", [
#     longformer_large], imdb_20news_3splits, standard_bigger_clusters_test)

# roberta_LOF_eval = EvalRun("roberta_LOF_eval", [
#     roberta_large], imdb_20news_3splits, small_transformer_test)

new_test = EvalRun("new_test", [
                   doc2vecwikiall_new, bert_base_uncased_new], imdb_20news_3splits, small_test_new)

# dictionary containing all the settings
eval_runs = {
    # "test_doc2vec": test_doc2vec, "full_doc2vec": full_doc2vec,
    #              "doc2vecwikiimdb20news013030": doc2vecwikiimdb20news013030, "test_transformer": test_transformer,
    #              "transformer_no_red": transformer_no_red, "longformer_eval": longformer_eval,
    #              "longformer_eval_big_clusters": longformer_eval_big_clusters,
    #              "longformer_large_bigger_clusters": longformer_large_bigger_clusters,
    #              "roberta_LOF_eval": roberta_LOF_eval,
    "new_test": new_test}

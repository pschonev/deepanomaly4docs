from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingModel:
    model_type: str  # to differentiate between doc2vec and transformer function
    model_name: str  # doc2vec, transformer name (BERT, RoBERTa,..) etc
    model_path: str
    model_train_data: str
    model_data_frac: float
    model_min_count: int
    model_epochs: int


@dataclass
class TestData:
    test_data_path: List[str]
    test_data_name: List[str]
    test_data_fraction: List[float]
    test_data_contamination: List[float]
    test_data_seed: List[int]


@dataclass
class TestSettings:
    n_comps: List[int]
    mix_ratio: List[float]
    umap_metric: List[str]
    min_cluster_size: List[int]
    allow_noise: List[bool]


@dataclass
class EvalRun:
    name: str
    model: List[EmbeddingModel]
    test_data: TestData
    test_settings: TestSettings
    res_folder: str = "/home/philipp/projects/dad4td/reports/clustering/"
    min_doc_length: int = 5


# model definitions
doc2vec011030 = EmbeddingModel(
    "doc2vec", "doc2vec", "/home/philipp/projects/dad4td/models/doc2vec_01_10/doc2vec_wiki.bin", "wikiEN", 0.1, 30, 10)
doc2vec013030 = EmbeddingModel(
    "doc2vec", "doc2vec", "/home/philipp/projects/dad4td/models/doc2vec_01_30/doc2vec_wiki.bin", "wikiEN", 0.1, 30, 30)


# test data settings
imdb_20news_3splits = TestData(["/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"], ["imdb_20news"], [0.15], [0.1], [42, 43, 44])


# evaluation settings
standard_test = TestSettings([3, 15, 45, 200], [0.0, 0.3, 0.15, 0.4], [
                             "cosine"], [15, 45, 90], [False])


# evaluation run definition
test_doc2vec = EvalRun("test_doc2vec", [
                       doc2vec011030, doc2vec013030], imdb_20news_3splits, standard_test)


# dictionary containing all the settings
eval_runs = {"test_doc2vec": test_doc2vec}

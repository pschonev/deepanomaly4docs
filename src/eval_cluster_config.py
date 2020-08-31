from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingModel:
    model_type: str  # to differentiate between doc2vec and transformer function
    model_name: str  # doc2vec, transformer name (BERT, RoBERTa,..) etc
    model_desc: str  # descriptive name
    model_path: str
    model_train_data: str
    model_data_frac: float
    model_epochs: int
    model_min_count: int


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
doc2vecwiki011030 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwiki011030", "/home/philipp/projects/dad4td/models/doc2vec_01_10/doc2vec_wiki.bin", "wikiEN", 0.1, 10, 30)
doc2vecwiki013030 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwiki013030", "/home/philipp/projects/dad4td/models/doc2vec_01_30/doc2vec_wiki.bin", "wikiEN", 0.1, 30, 30)
doc2vecwiki013001 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwiki013001", "/home/philipp/projects/dad4td/models/doc2vec_01_30_min1/doc2vec_wiki.bin", "wikiEN", 0.1, 30, 1)
doc2vecwiki031030 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwiki031030", "/home/philipp/projects/dad4td/models/doc2vec_03_10/doc2vec_wiki.bin", "wikiEN", 0.3, 10, 30)
doc2vecimdb20news101001 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecimdb20news101001", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_10_min1/doc2vec_wiki.bin", "imdb_20news", 1.0, 10, 1)
doc2vecimdb20news1010001 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecimdb20news1010001", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_100_min1/doc2vec_wiki.bin", "imdb_20news", 1.0, 100, 1)
doc2vecwikiimdb20news011001 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwikiimdb20news011001", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min1/doc2vec_wiki.bin", "wiki_imdb_20news", 0.1, 10, 1)
doc2vecwikiimdb20news011030 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwikiimdb20news011030", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min30/doc2vec_wiki.bin", "wiki_imdb_20news", 0.1, 10, 30)
doc2vecwikiimdb20news013030 = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwikiimdb20news013030", "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min30/doc2vec_wiki.bin", "wiki_imdb_20news", 0.1, 30, 30)
doc2vecapnews = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecapnews", "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin", "apnews", 1.0, 100, 1)
doc2vecwikiall = EmbeddingModel(
    "doc2vec", "doc2vec", "doc2vecwikiall", "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin", "wikiEN", 1.0, 100, 1)


# test data settings
imdb_20news_3splits = TestData(
    ["/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"], ["imdb_20news"], [0.15], [0.1], [42, 43, 44])


# evaluation settings
standard_test = TestSettings([3, 6, 15, 45, 200], [0.0, 0.15, 0.3, 0.4, 1.0], [
                             "cosine"], [15, 45, 90], [False])


# evaluation run definition
test_doc2vec = EvalRun("test_doc2vec", [
                       doc2vecwiki011030, doc2vecwiki013030], imdb_20news_3splits, standard_test)

full_doc2vec = EvalRun("full_doc2vec", [
                       doc2vecwiki011030, doc2vecwiki013030, doc2vecwiki013001, doc2vecwiki031030, doc2vecimdb20news101001,
                       doc2vecimdb20news1010001, doc2vecwikiimdb20news011001, doc2vecwikiimdb20news011030, doc2vecapnews, doc2vecwikiall
                       ], imdb_20news_3splits, standard_test)
doc2vecwikiimdb20news013030 = EvalRun("doc2vecwikiimdb20news013030", [
    doc2vecwikiimdb20news013030], imdb_20news_3splits, standard_test)

# dictionary containing all the settings
eval_runs = {"test_doc2vec": test_doc2vec, "full_doc2vec": full_doc2vec,
             "doc2vecwikiimdb20news013030": doc2vecwikiimdb20news013030}

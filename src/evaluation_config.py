from gensim.models.doc2vec import Doc2Vec
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.knn import KNN
from sklearn.decomposition import PCA as PCAR
from sklearn.manifold import TSNE
from ivis import Ivis
from umap import UMAP
from evaluation import TestData, Doc2VecModel, WordEmbeddingPooling, RNNEmbedding, TransformerModel, NoReduction, SklearnReducer, PyodDetector, DimRedOutlierDetector, HDBSCAN_GLOSH, EvalRun


# vectorize model
doc2vecwikiall = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                              100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")
doc2vecapnews = Doc2VecModel("doc2vecapnews", "apnews", 1.0,
                             100, 1, "/home/philipp/projects/dad4td/models/apnews_dbow/doc2vec.bin")

doc2vecwiki011030 = Doc2VecModel("doc2vecwiki011030", "wiki_EN", 0.1,
                                 10, 30, "/home/philipp/projects/dad4td/models/doc2vec_01_10_30/doc2vec_wiki.bin")
doc2vecwiki013030 = Doc2VecModel("doc2vecwiki013030", "wiki_EN", 0.1,
                                 30, 30, "/home/philipp/projects/dad4td/models/doc2vec_01_30_30/doc2vec_wiki.bin")
doc2vecwiki013001 = Doc2VecModel("doc2vecwiki013001", "wiki_EN", 0.1,
                                 30, 1, "/home/philipp/projects/dad4td/models/doc2vec_01_30_1/doc2vec_wiki.bin")
doc2vecwiki031030 = Doc2VecModel("doc2vecwiki031030", "wiki_EN", 0.3,
                                 10, 30, "/home/philipp/projects/dad4td/models/doc2vec_03_10_30/doc2vec_wiki.bin")

doc2vecimdb20news101001 = Doc2VecModel("doc2vecimdb20news101001", "imdb_20news", 1.0,
                                       10, 1, "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_10_min1/doc2vec_wiki.bin")
doc2vecimdb20news1010001 = Doc2VecModel("doc2vecimdb20news1010001", "imdb_20news", 1.0,
                                        100, 1, "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_100_min1/doc2vec_wiki.bin")

doc2vecwikiimdb20news011001 = Doc2VecModel("doc2vecwikiimdb20news011001", "wiki_EN_imdb_20news", 0.1,
                                           10, 1, "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min1/doc2vec_wiki.bin")
doc2vecwikiimdb20news011030 = Doc2VecModel("doc2vecwikiimdb20news011030", "wiki_EN_imdb_20news", 0.1,
                                           10, 30, "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_10_min30/doc2vec_wiki.bin")
doc2vecwikiimdb20news013030 = Doc2VecModel("doc2vecwikiimdb20news013030", "wiki_EN_imdb_20news", 0.1,
                                           30, 30, "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_wiki_01_30_min30/doc2vec_wiki.bin")

allnews_05_15_30 = Doc2VecModel("allnews_05_15_30", "all_news", 0.5,
                                15, 30, "/home/philipp/projects/dad4td/models/all_news_05_30_30/all_news.bin")


all_doc2vec = [doc2vecwikiall, doc2vecapnews, doc2vecwiki011030, doc2vecwiki013030, doc2vecwiki013001, doc2vecwiki031030,
               doc2vecimdb20news101001, doc2vecimdb20news1010001, doc2vecwikiimdb20news011001, doc2vecwikiimdb20news011030, doc2vecwikiimdb20news013030]

longformer_large = TransformerModel(
    "allenai/longformer-large-4096", "long_documents_large", 435)

longformer_base = TransformerModel(
    "allenai/longformer-base-4096", "long_documents", 149)

glove = WordEmbeddingPooling("glove", "unknown")

glove_trec_6 = RNNEmbedding(
    "glove_trec_6", "trec_6", "/home/philipp/projects/dad4td/models/glove_trec_6/best-model.pt")

glove_amazon = RNNEmbedding(
    "glove_amazon", "amazon", "/home/philipp/projects/dad4td/models/glove_amazon/best-model.pt")

fasttext_amazon = RNNEmbedding(
    "fasttext_amazon", "amazon", "/home/philipp/projects/dad4td/models/fasttext-amazon/best-model.pt")

# test data
imdb_20news_3splits = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.15], contamination=[0.1], seed=[43, 44, 42])
imdb_20news_3splits_small = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.05], contamination=[0.1], seed=[42, 43, 44])
imdb_20news_3splits_verysmall = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.01], contamination=[0.1], seed=[42, 43, 44])
imdb_20news_3splits_full = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[1.0], contamination=[0.1], seed=[42, 43, 44])
imdb_20news_3split_fracs = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.01, 0.05, 0.1], contamination=[0.1], seed=[42, 43, 44])
imdb_20news_3split_fracs_med = TestData(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl", "imdb_20news", fraction=[0.05, 0.1, 0.15], contamination=[0.1], seed=[42, 43, 44])


# dimension reduction
umap_all = SklearnReducer(UMAP, "UMAP", dict(
    n_components=[2, 4, 8, 16, 64, 256], set_op_mix_ratio=[0.5, 1.0], metric=["cosine"]))
tsne_all = SklearnReducer(TSNE, "TSNE", dict(
    n_components=[3], perplexity=[10, 30, 45]))
ivis = SklearnReducer(Ivis, "ivis", dict(embedding_dims=[2, 4, 8, 16, 64, 256], k=[15],
                                         n_epochs_without_progress=[15], model=["model"], metric=["pn", "cosine"]))


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
                           [tsne_all],
                           [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                            PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA"), glosh_test])

pyod_test_umap_all_data = EvalRun("pyod_test_umap_all_data",
                                  [doc2vecwikiall],
                                  [imdb_20news_3splits_full],
                                  [],
                                  [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                                      PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

pyod_test_umap_all_data_no_red = EvalRun("pyod_test_umap_all_data_no_red",
                                         [longformer_large],
                                         [imdb_20news_3splits_full],
                                         [NoReduction()],
                                         [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                                             PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

pyod_umap_big_dim = EvalRun("pyod_umap_big_dim",
                            [doc2vecwikiall],
                            [imdb_20news_3splits],
                            [],
                            [PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                             PyodDetector(LOF, "LOF"), PyodDetector(OCSVM, "OCSVM"), PyodDetector(PCA, "PCA")])

pyod_autoencoder_test = EvalRun("pyod_autoencoder_test",
                                [doc2vecwikiall, longformer_large],
                                [imdb_20news_3splits],
                                [NoReduction()],
                                [PyodDetector(VAE(epochs=30, verbosity=1), "VAE_30"),
                                 PyodDetector(
                                     VAE(epochs=100, verbosity=1), "VAE_100"),
                                 PyodDetector(AutoEncoder(
                                     epochs=30, verbose=1), "AE_30"),
                                 PyodDetector(AutoEncoder(epochs=100, verbose=2), "AE_100")])

pyod_autoencer_refined = EvalRun("pyod_autoencer_refined",
                                 [doc2vecwikiall, doc2vecapnews],
                                 [imdb_20news_3split_fracs],
                                 [],
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
                                       [],
                                       [PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                                 epochs=10, verbose=1), "AE_10_tiny"),
                                        PyodDetector(AutoEncoder(hidden_neurons=[16, 8, 8, 16],
                                                                 epochs=30, verbose=1), "AE_30_tiny")
                                        ])

pyod_autoencer_refined_ext = EvalRun("pyod_autoencer_refined_ext",
                                     [doc2vecapnews],
                                     [imdb_20news_3split_fracs_med],
                                     [],
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
                                   [NoReduction()],
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
                                         [],
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
                                                                 [doc2vecapnews,
                                                                     doc2vecwikiall],
                                                                 [imdb_20news_3splits_full],
                                                                 [NoReduction()],
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

test_doc2vec_models = EvalRun("test_doc2vec_models",
                              all_doc2vec,
                              [imdb_20news_3splits],
                              [],
                              [
                                  PyodDetector(HBOS, "HBOS"),
                                  PyodDetector(OCSVM, "OCSVM"),
                                  PyodDetector(AutoEncoder, "AE", dict(
                                      hidden_neurons=[2, 1, 2], epochs=1)),
                                  PyodDetector(AutoEncoder, "AE", dict(
                                      hidden_neurons=[2, 1, 2], epochs=3))
                              ])

test_wordemb = EvalRun("test_wordemb",
                       [glove],
                       [imdb_20news_3splits],
                       [NoReduction()],
                       [
                           PyodDetector(HBOS, "HBOS"),
                           PyodDetector(OCSVM, "OCSVM"),
                           PyodDetector(AutoEncoder, "AE", dict(
                               hidden_neurons=[2, 1, 2], epochs=1)),
                           PyodDetector(AutoEncoder, "AE", dict(
                               hidden_neurons=[2, 1, 2], epochs=3))
                       ])

test_wordemb_2 = EvalRun("test_wordemb_2",
                         [WordEmbeddingPooling("news", "news_wiki"),
                          WordEmbeddingPooling("extvec", "unknown"),
                          WordEmbeddingPooling("crawl", "unknown"),
                          WordEmbeddingPooling("twitter", "twitter"),
                          WordEmbeddingPooling("turian", "unknown"),
                          WordEmbeddingPooling("glove", "unknown")],
                         [imdb_20news_3splits_small],
                         [NoReduction()],
                         [
                             PyodDetector(HBOS, "HBOS"),
                             PyodDetector(OCSVM, "OCSVM"),
                             PyodDetector(AutoEncoder, "AE", dict(
                                 hidden_neurons=[2, 1, 2], epochs=1)),
                             PyodDetector(AutoEncoder, "AE", dict(
                                 hidden_neurons=[2, 1, 2], epochs=3))
                         ])

test_wordemb_rnn = EvalRun("test_wordemb_rnn",
                           [glove_trec_6],
                           [imdb_20news_3splits_small],
                           [NoReduction()],
                           [
                               PyodDetector(HBOS, "HBOS"),
                               PyodDetector(OCSVM, "OCSVM"),
                               PyodDetector(AutoEncoder, "AE", dict(
                                   hidden_neurons=[2, 1, 2], epochs=1)),
                               PyodDetector(AutoEncoder, "AE", dict(
                                   hidden_neurons=[2, 1, 2], epochs=3))
                           ])

test_wordemb_rnn_amazon = EvalRun("test_wordemb_rnn_amazon",
                                  [glove_amazon],
                                  [imdb_20news_3splits_small],
                                  [NoReduction()],
                                  [
                                      PyodDetector(HBOS, "HBOS"),
                                      PyodDetector(OCSVM, "OCSVM"),
                                      PyodDetector(AutoEncoder, "AE", dict(
                                          hidden_neurons=[2, 1, 2], epochs=1)),
                                      PyodDetector(AutoEncoder, "AE", dict(
                                          hidden_neurons=[2, 1, 2], epochs=3))
                                  ])

test_wordemb_rnn_amazon_fasttext = EvalRun("test_wordemb_rnn_amazon_fasttext",
                                           [fasttext_amazon],
                                           [imdb_20news_3splits_small],
                                           [SklearnReducer(UMAP, "UMAP", dict(
                                               n_components=[2, 64, 128, 256], set_op_mix_ratio=[1.0], metric=["cosine"])),
                                            NoReduction()],
                                           [
                                               PyodDetector(AutoEncoder, "AE", dict(
                                                   hidden_neurons=[2, 1, 2], epochs=100)),
                                               PyodDetector(AutoEncoder, "AE", dict(
                                                   hidden_neurons=[8, 4, 2, 4, 8], epochs=100))
                                           ])

ivis_test = EvalRun("ivis_test",
                    [doc2vecapnews,
                     doc2vecwikiall],
                    [imdb_20news_3splits],
                    [SklearnReducer(Ivis, "ivis", dict(embedding_dims=[16, 64, 256], k=[15],
                                                       n_epochs_without_progress=[15, 30], model=["maaten"], distance=["pn", "cosine"])),
                     NoReduction()],
                    [
                        PyodDetector(HBOS, "HBOS"),
                        PyodDetector(OCSVM, "OCSVM"),
                        PyodDetector(AutoEncoder, "AE", dict(
                            hidden_neurons=[2, 1, 2], epochs=1)),
                        PyodDetector(AutoEncoder, "AE", dict(
                            hidden_neurons=[2, 1, 2], epochs=3))
                    ]
                    )

dimred_dimred = EvalRun("dimred_dimred",
                        [doc2vecapnews,
                         doc2vecwikiall],
                        [imdb_20news_3splits],
                        [SklearnReducer(UMAP, "UMAP", False, dict(
                            n_components=[64, 128, 256, 300], set_op_mix_ratio=[1.0], metric=["cosine"])),
                            NoReduction()],
                        [DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[15],
                                                                        n_epochs_without_progress=[40, 15], model=["maaten"], distance=["pn"])),
                         ]
                        )

dimred_dimred_big = EvalRun("dimred_dimred_big",
                            [
                                doc2vecwiki011030,
                                doc2vecwiki013030,
                                doc2vecimdb20news1010001,
                                doc2vecwikiimdb20news013030,
                                doc2vecapnews,
                                doc2vecwikiall,
                            ],
                            [imdb_20news_3splits],
                            [SklearnReducer(UMAP, "UMAP", False, dict(
                                n_components=[2, 8, 64, 128, 256, 300], set_op_mix_ratio=[1.0], metric=["cosine"])),
                                NoReduction()],
                            [DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[15],
                                                                            n_epochs_without_progress=[20], model=["maaten"], distance=["pn"])),
                             ]
                            )

dimred_dimred_big_2 = EvalRun("dimred_dimred_big_2",
                              [
                                  WordEmbeddingPooling("extvec", "unknown"),
                                  WordEmbeddingPooling("glove", "unknown"),
                              ],
                              [imdb_20news_3splits_small],
                              [SklearnReducer(UMAP, "UMAP", False, dict(
                                  n_components=[2, 8, 64, 128, 256, 300], set_op_mix_ratio=[1.0], metric=["cosine"])),
                                  NoReduction()],
                              [DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[15],
                                                                              n_epochs_without_progress=[20], model=["maaten"], distance=["pn"])),
                               ]
                              )

dimred_dimred_longf = EvalRun("dimred_dimred_longf",
                              [
                                  longformer_large
                              ],
                              [imdb_20news_3splits_verysmall],
                              [SklearnReducer(UMAP, "UMAP", False, dict(
                                  n_components=[256], set_op_mix_ratio=[1.0], metric=["cosine"])),
                                  NoReduction()],
                              [DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[15],
                                                                              n_epochs_without_progress=[20], model=["maaten"], distance=["pn"])),
                               ]
                              )

all_news_test = EvalRun("all_news_test",
                        [allnews_05_15_30],
                        [imdb_20news_3splits],
                        [SklearnReducer(UMAP, "UMAP", False, dict(
                            n_components=[2, 8, 64, 128, 256, 300], set_op_mix_ratio=[1.0], metric=["cosine"])),
                         NoReduction()],
                        [PyodDetector(AutoEncoder, "AE", dict(
                            hidden_neurons=[[2, 1, 2]], epochs=[1])),
                         PyodDetector(HBOS, "HBOS"),
                         PyodDetector(OCSVM, "OCSVM"),
                         DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[15],
                                                                        n_epochs_without_progress=[20], model=["maaten"], distance=["pn"]))
                         ])

invis_params = EvalRun("invis_params",
                       [allnews_05_15_30],
                       [imdb_20news_3splits],
                       [SklearnReducer(UMAP, "UMAP", False, dict(
                           n_components=[300, 8], set_op_mix_ratio=[1.0], metric=["cosine"]))],
                       [DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[5, 15, 50, 150, 300],
                                                                       n_epochs_without_progress=[2, 5, 15, 25, 50], model=["maaten", "szubert", "hinton"], distance=["pn"]))
                        ])

all_new_mono = EvalRun("all_new_mono",
                       [allnews_05_15_30],
                       [imdb_20news_3splits],
                       [SklearnReducer(UMAP, "UMAP", False, dict(
                           n_components=[1], set_op_mix_ratio=[1.0], metric=["cosine"]))],
                       [PyodDetector(AutoEncoder, "AE", dict(
                           hidden_neurons=[[1]], epochs=[1]))
                        ])


KNN_and_ivis = EvalRun("KNN_and_ivis",
                       [doc2vecwikiall, doc2vecapnews,
                        doc2vecwikiimdb20news011001, longformer_large],
                       [imdb_20news_3splits],
                       [SklearnReducer("UMAP", UMAP, False, dict(
                           n_components=[3, 6, 15, 50, 100, 200, 300],
                           set_op_mix_ratio=[1.0], metric=["cosine"])), NoReduction()],
                       [PyodDetector(KNN, "KNN"),
                        DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[5],
                                                                       n_epochs_without_progress=[
                                                                           5],
                                                                       model=["maaten"], distance=["pn"])),
                        ])

KNN_and_ivis_longformer = EvalRun("KNN_and_ivis_longformer",
                                  [longformer_base],
                                  [imdb_20news_3splits],
                                  [SklearnReducer("UMAP", UMAP, False, dict(
                                      n_components=[
                                          3, 6, 15, 50, 100, 200, 300],
                                      set_op_mix_ratio=[1.0], metric=["cosine"])), NoReduction()],
                                  [PyodDetector(KNN, "KNN"),
                                      DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[5],
                                                                                     n_epochs_without_progress=[
                                                                                         5],
                                                                                     model=["maaten"], distance=["pn"])),
                                   ])

all_news_all = EvalRun("all_news_all",
                       [allnews_05_15_30],
                       [imdb_20news_3splits],
                       [SklearnReducer("UMAP", UMAP, False, dict(
                           n_components=[3, 6, 15, 50, 100, 200, 300],
                           set_op_mix_ratio=[1.0], metric=["cosine"])), NoReduction()],
                       [PyodDetector(KNN, "KNN"), PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                        PyodDetector(LOF, "LOF"), PyodDetector(
                           OCSVM, "OCSVM"), PyodDetector(PCA, "PCA"),
                        DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[5],
                                                                       n_epochs_without_progress=[
                                                                           5],
                                                                       model=["maaten"], distance=["pn"])),
                        ])

AE_all = EvalRun("AE_all",
                       [doc2vecwikiall, doc2vecapnews,
                        doc2vecwikiimdb20news011001, allnews_05_15_30, longformer_large],
                       [imdb_20news_3splits],
                       [SklearnReducer("UMAP", UMAP, False, dict(
                           n_components=[3, 6, 15, 50, 100, 200, 300],
                           set_op_mix_ratio=[1.0], metric=["cosine"])), NoReduction()],
                       [PyodDetector(AutoEncoder, "AE", dict(
                            hidden_neurons=[[2, 1, 2]], epochs=[3]))
                        ])

WEP_RNNs = EvalRun("WEP_RNNs",
                       [glove, glove_trec_6, glove_amazon, fasttext_amazon],
                       [imdb_20news_3splits],
                       [SklearnReducer("UMAP", UMAP, False, dict(
                           n_components=[3, 6, 15, 50, 100, 200, 300],
                           set_op_mix_ratio=[1.0], metric=["cosine"])), NoReduction()],
                       [PyodDetector(KNN, "KNN"), PyodDetector(HBOS, "HBOS"), PyodDetector(IForest, "iForest"),
                        PyodDetector(LOF, "LOF"), PyodDetector(
                           OCSVM, "OCSVM"), PyodDetector(PCA, "PCA"),
                        DimRedOutlierDetector(Ivis, "ivis", True, dict(embedding_dims=[1], k=[5],
                                                                       n_epochs_without_progress=[
                                                                           5],
                                                                       model=["maaten"], distance=["pn"])),
                        PyodDetector(AutoEncoder, "AE", dict(
                            hidden_neurons=[[2, 1, 2]], epochs=[3]))
                        ])

AE_longformer = EvalRun("AE_longformer",
                       [longformer_base],
                       [imdb_20news_3splits],
                       [SklearnReducer("UMAP", UMAP, False, dict(
                           n_components=[3, 6, 15, 50, 100, 200, 300],
                           set_op_mix_ratio=[1.0], metric=["cosine"])), NoReduction()],
                       [PyodDetector(AutoEncoder, "AE", dict(
                            hidden_neurons=[[2, 1, 2]], epochs=[3]))
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
    "pyod_autoencoder_mono": pyod_autoencoder_mono,
    "test_doc2vec_models": test_doc2vec_models,
    "test_wordemb": test_wordemb,
    "test_wordemb_2": test_wordemb_2,
    "test_wordemb_rnn": test_wordemb_rnn,
    "test_wordemb_rnn_amazon": test_wordemb_rnn_amazon,
    "test_wordemb_rnn_amazon_fasttext": test_wordemb_rnn_amazon_fasttext,
    "ivis_test": ivis_test,
    "dimred_dimred": dimred_dimred,
    "dimred_dimred_big": dimred_dimred_big,
    "dimred_dimred_big_2": dimred_dimred_big_2,
    "dimred_dimred_longf": dimred_dimred_longf,
    "all_news_test": all_news_test,
    "invis_params": invis_params,
    "KNN_and_ivis": KNN_and_ivis,
    "KNN_and_ivis_longformer": KNN_and_ivis_longformer,
    "all_news_all": all_news_all,
    "AE_all": AE_all,
    "WEP_RNNs": WEP_RNNs,
    "AE_longformer": AE_longformer
    }


def main():
    pass


if __name__ == "__main__":
    main()

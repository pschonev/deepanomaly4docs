# python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as g
import logging
from umap import UMAP
from ivis import Ivis
from collections import defaultdict
import numpy as np
import pandas as pd
#from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from evaluation import get_scores, reject_outliers, sample_data

tqdm.pandas(desc="progess: ")

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, save_path):
        self.save_path = save_path

    def on_epoch_end(self, model):
        #output_path = get_tmpfile(self.save_path)
        model.save(self.save_path)
        print("epoch model saved to {self.save_path}...")


class EpochResult(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, test_data, log_path):
        self.test_data = test_data
        self.log_path = log_path

        self.result_df = pd.DataFrame()
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(
            f"\n----------------\n\nEnd of epoch {self.epoch}. Getting scores...")
        scores = defaultdict(list)
        scores["epoch"] = self.epoch
        for df, seed in test_data:
            print(f"Vectorize...")

            docvecs = df["text"].progress_apply(lambda x: simple_preprocess(x))
            docvecs = docvecs.progress_apply(lambda x: model.infer_vector(x))
            
            print(f"Reduce dimensions...")
            dim_reducer = UMAP(metric="cosine", set_op_mix_ratio=1.0,
                            n_components=256, random_state=42)

            dim_reduced_vecs = dim_reducer.fit_transform(list(docvecs))

            print(f"Run ivis...")
            dim_reducer = Ivis(embedding_dims=1, k=15,
                            model="maaten", n_epochs_without_progress=10, verbose=0)
            decision_scores = dim_reducer.fit_transform(dim_reduced_vecs)
            decision_scores = decision_scores.astype(float)

            print(f"Get and save scores...")
            preds = reject_outliers(decision_scores, iq_range=1.0-contamination)
            preds = [-1 if x else 1 for x in preds]

            
            scores = get_scores(scores, df["outlier_label"], preds)
            scores["seed"] = seed
            print(f"Scores for epoch {self.epoch} | seed - {seed}:\n{pd.DataFrame(scores, index=[0])}")

            self.result_df = self.result_df.append(
                                    scores, ignore_index=True)
            self.result_df.to_csv(self.log_path, sep="\t")
        self.epoch += 1


# test data
seeds = [42, 43]
fraction = 1.0
contamination = 0.1

test_data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl"
df = pd.read_pickle(test_data_path)
test_data = [(sample_data(df, fraction, contamination, seed), seed) for seed in seeds]
#test_data = test_data + [(sample_data(df, fraction=1.0, contamination=contamination, seed=1), 1)]

# doc2vec parameters
vector_size = 300
window_size = 15
min_count = 20
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 50
dm = 0  # 0 = dbow; 1 = dmpv
worker_count = 4  # number of parallel processes

# pretrained word embeddings
# None if use without pretrained embeddings
pretrained_emb = "/home/philipp/projects/dad4td/models/en_wiki_w2vec/word2vec.bin"
# pretrained_emb = None

# input corpus
train_corpus = "/home/philipp/projects/dad4td/data/raw/all-the-news-2_1_05_sf.txt"

# save folder
save_folder = "/home/philipp/projects/dad4td/models/all_news_05_50_20"

# output model
save_path = save_folder + "/all_news.bin"

# mapfile
mapfile = save_folder + "/all_news_mapfile.txt"

# log_path
log_path = save_folder + "/all_news_results.tsv"

# enable logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0,
                  dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch, docvecs_mapfile=mapfile,
                  callbacks=[EpochSaver(save_path=save_path), EpochResult(test_data=test_data, log_path=log_path)])

# save model
model.save(save_path)

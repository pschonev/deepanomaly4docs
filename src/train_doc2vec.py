# python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as g
import logging

# doc2vec parameters
vector_size = 300
window_size = 15
min_count = 30
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 30
dm = 0  # 0 = dbow; 1 = dmpv
worker_count = 4  # number of parallel processes

# pretrained word embeddings
pretrained_emb = "/home/philipp/projects/dad4td/models/en_wiki_w2vec/word2vec.bin" #None if use without pretrained embeddings
#pretrained_emb = None

# input corpus
train_corpus = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_apnews_01.txt"

# output model
saved_path = "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_apnews_01_30_min30/doc2vec_apnews.bin"

# mapfile
mapfile = "/home/philipp/projects/dad4td/models/doc2vec_20_news_imdb_apnews_01_30_min30/doc2vec_apnews_mapfile.txt"

# enable logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0,
                  dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch, docvecs_mapfile=mapfile)

# save model
model.save(saved_path)

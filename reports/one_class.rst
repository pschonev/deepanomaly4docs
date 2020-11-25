One class classification
=========================================

**Learning Deep Features for One-Class Classification**

https://arxiv.org/abs/1801.05365

- take one network and train it with two different outputs and losses on different data

- the first takes inlier data and tries to reduce variance on the output vectors (compactness)

- the second gets outlier data with class labels and tries to optimize categorical crossentropy (distinctiveness

**performance**

- similar to supervised in "supervised" mode

- using 20 newsgroup as outliers it learns something but not very good

- combining them doesn't do anything

- possibly a bit better with less data than standard supervised

- not tested with actualy finetuning of bigger network to get embeddings

- not tested with fully connected on top of network trained with the above training regime
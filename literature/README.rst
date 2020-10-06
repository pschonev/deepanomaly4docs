Ressources
=========================================

Literature
-----------

Ruff et. al.
^^^^^^^^^^^^
- Ruff, Lukas, Robert Vandermeulen, et al. **“Deep One-Class Classification.”** International Conference on Machine Learning, 2018, pp. 4393–402. proceedings.mlr.press, http://proceedings.mlr.press/v80/ruff18a.html.
- Ruff, Lukas, Robert A. Vandermeulen, Nico Görnitz, et al. **“Deep Semi-Supervised Anomaly Detection.”** ArXiv:1906.02694 [Cs, Stat], Feb. 2020. arXiv.org, http://arxiv.org/abs/1906.02694.
- Ruff, Lukas, Robert A. Vandermeulen, Billy Joe Franks, et al. **“Rethinking Assumptions in Deep Anomaly Detection.”** ArXiv:2006.00339 [Cs, Stat], May 2020. arXiv.org, http://arxiv.org/abs/2006.00339.
- Ruff, Lukas, Yury Zemlyanskiy, et al. **“Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text.”** Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, 2019, pp. 4061–71. DOI.org (Crossref), doi:10.18653/v1/P19-1398.

Outlier Exposure
^^^^^^^^^^^^^^^^^

- Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich. **“Deep Anomaly Detection with Outlier Exposure.”** ArXiv:1812.04606 [Cs, Stat], Jan. 2019. arXiv.org, http://arxiv.org/abs/1812.04606.
- Hendrycks, Dan, and Kevin Gimpel. **“A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.”** ArXiv:1610.02136 [Cs], Oct. 2018. arXiv.org, http://arxiv.org/abs/1610.02136.

Deep Methods
^^^^^^^^^^^^^

- Pang, Guansong, et al. **“Deep Anomaly Detection with Deviation Networks.”** ArXiv:1911.08623 [Cs, Stat], Nov. 2019. arXiv.org, http://arxiv.org/abs/1911.08623.
- Pang, Guansong, et al. **“Deep Weakly-Supervised Anomaly Detection.”** ArXiv:1910.13601 [Cs, Stat], Jan. 2020. arXiv.org, http://arxiv.org/abs/1910.13601.

|
|

- Golan, Izhak, and Ran El-Yaniv. **“Deep Anomaly Detection Using Geometric Transformations.”** ArXiv:1805.10917 [Cs, Stat], Nov. 2018. arXiv.org, http://arxiv.org/abs/1805.10917.
- Hendrycks, Dan, Mantas Mazeika, Saurav Kadavath, et al. **Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty.** p. 12.

Autoencoder
^^^^^^^^^^^^

- Huang, Chaoqin, et al. **“Attribute Restoration Framework for Anomaly Detection.”** ArXiv:1911.10676 [Cs], June 2020. arXiv.org, http://arxiv.org/abs/1911.10676.
- Cao, Van Loi, et al. **“A Hybrid Autoencoder and Density Estimation Model for Anomaly Detection.”** Parallel Problem Solving from Nature – PPSN XIV, edited by Julia Handl et al., vol. 9921, Springer International Publishing, 2016, pp. 717–26. DOI.org (Crossref), doi:10.1007/978-3-319-45823-6_67.
- Schreyer, Marco, et al. **“Detection of Anomalies in Large Scale Accounting Data Using Deep Autoencoder Networks.”** ArXiv:1709.05254 [Cs], Aug. 2018. arXiv.org, http://arxiv.org/abs/1709.05254.

Doc2Vec
^^^^^^^^

- Le, Quoc V., and Tomas Mikolov. **“Distributed Representations of Sentences and Documents.”** ArXiv:1405.4053 [Cs], May 2014. arXiv.org, http://arxiv.org/abs/1405.4053.
- Lau, Jey Han, and Timothy Baldwin. **“An Empirical Evaluation of Doc2vec with Practical Insights into Document Embedding Generation.”** ArXiv:1607.05368 [Cs], July 2016. arXiv.org, http://arxiv.org/abs/1607.05368.

UMAP
^^^^^

- McInnes, Leland, et al. **“UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.”** ArXiv:1802.03426 [Cs, Stat], Dec. 2018. arXiv.org, http://arxiv.org/abs/1802.03426.
- Allaoui, Mebarka, et al. **“Considerably Improving Clustering Algorithms Using UMAP Dimensionality Reduction Technique: A Comparative Study.”** Image and Signal Processing, edited by Abderrahim El Moataz et al., Springer International Publishing, 2020, pp. 317–25. Springer Link, doi:10.1007/978-3-030-51935-3_34.
- Sainburg, Tim, et al. **“Parametric UMAP: Learning Embeddings with Deep Neural Networks for Representation and Semi-Supervised Learning.”** ArXiv:2009.12981 [Cs, q-Bio, Stat], Sept. 2020. arXiv.org, http://arxiv.org/abs/2009.12981.

Code
-----

- Uniform Manifold Approximation and Projection (UMAP) - https://github.com/lmcinnes/umap
- Python Outlier Detection (PyOD) - https://github.com/yzhao062/pyod
- flair - https://github.com/flairNLP/flair (for word embedding pooling, RNNs and transformer embeddings)
- gensim - https://radimrehurek.com/gensim/index.html (Doc2Vec)
- ivis - https://bering-ivis.readthedocs.io/en/latest/ (siamese network dimensionality reduction used as outlier detector)


Data
-----

- Training doc2vec: All the news https://components.one/datasets/all-the-news-2-news-articles-dataset/
- Inlier data: IMDB Reviews https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Outlier data: 20 Newsgroup http://qwone.com/~jason/20Newsgroups/
- pretrained doc2vec models: https://github.com/jhlau/doc2vec (see Lau, Jey Han, and Timothy Baldwin above)

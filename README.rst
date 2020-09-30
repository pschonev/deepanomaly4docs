Deep Anomaly Detection for Text Documents
=========================================

Plan
=====
- using test data (IMDB / 20 Newsgroup)
- ☑ TF-IDF Embeddings with UMAP and LOF
- ☑ Visualization
- ☑ enstop topic modelling
- ☑ HDBSCAN cluster analysis and outlier detection (GLOSH) https://hdbscan.readthedocs.io/en/latest/outlier_detection.html
- ☑ outlier detection algorithms from PyOD (LOF, HBOS, PCA, IForest) https://github.com/yzhao062/pyod

- ☑ test flair https://github.com/flairNLP/flair
- ☑ Transformer embeddings
- ☑ word embedding pooling - word2vec, glove, fasttext
- ☑ word embedding RNN/LSTM
- ☐ Autoencoder embeddings

|

- ☑ Autoencoder loss (with progress on outlier f1)
- ☑ Siamese Network (ivis)
- ☐ other new DL approaches

|

- ☐ Everything above but on real data
 
|

- ☐ unsupervised vs weakly supervised
- ☐ ensembles
- ☐ Compare with computer vision approach

|

Currently: 

|

- monitoring progress on doc2vec training
- test if ivis unstable over runs 


Detect outliers in text document datasets.

Project Organization
=====================

::

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

import umap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

feat_folder_p = Path("/home/philipp/projects/dad4td/data/processed/")


def get_feat_path(data_path, suffix=""):
    data_name = Path(data_path)
    return feat_folder_p / f'{data_name.stem}{suffix}.npy'


def load_data(path, dropna=True):
    """ Check if csv, tsv or pkl and load data to dataframe
    """
    print(f"Load data from {path} ...")
    df = pd.read_csv(path)
    if dropna:
        df = df.dropna()
    return df


def get_tf_idf(df, col):
    """Get tf-idf from pandas dataframe text column. Removes stop words.

    :param df: The dataframe containing the text
    :type df: pandas dataframe object
    :param col: Name of the text column
    :type col: string
    :return: Numpy array containing the tf-idf
    :rtype: [type]
    """
    print("Get TF-IDF vectors ...")
    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    return tfidf_vectorizer.fit_transform(df[col])


def loadcreate_umap_emb(tfidf_word_doc_matrix, path, load=True, umap_kwargs={}):
    if Path(path).is_file() and load:
        print("Loading UMAP embeddings ...")
        tfidf_embedding = np.load(path)
    else:
        np.save(path, [])
        print("Create UMAP embeddings ...")
        tfidf_embedding = umap.UMAP(
            **umap_kwargs).fit_transform(tfidf_word_doc_matrix)
        np.save(path, tfidf_embedding)

    return tfidf_embedding

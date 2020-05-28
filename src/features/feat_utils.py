import umap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def get_new_path(file_path, new_folder, file_ext, suffix="", proj_folder="/home/philipp/projects/dad4td/"):
    proj_folder = Path(proj_folder)
    file_path = Path(file_path)
    return proj_folder / new_folder / f'{file_path.stem}{suffix}.{file_ext}'


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

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


def loadcreate(input_embeddings, path, model, load=True, save=True):
    if Path(path).is_file() and load:
        print(f"Loading embeddings from {path} ...")
        output_embeddings = np.load(path)
    else:
        np.save(path, [])
        print(f"Creating embeddings using {model}...")
        output_embeddings = model.fit_transform(input_embeddings)
        if save:
            np.save(path, output_embeddings)
        print("Embeddings created")

    return output_embeddings

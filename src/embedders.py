from abc import abstractmethod, ABC
from pydantic import BaseModel
from typing import Any
import numpy as np
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec
from flair.embeddings import TransformerDocumentEmbeddings, WordEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from flair.data import Sentence


# model for conversion from text to vectors
class EmbeddingModel(BaseModel, ABC):
    """Abstract class to vectorize text."""
    # doc2vec, or huggingface transformer specifier (e.g. bert-uncased)
    model_name: str
    model_train_data: str
    text_col: str = "text"

    @abstractmethod
    def vectorize(self, X, data_col=None):
        pass


class Doc2VecModel(EmbeddingModel):
    """Loads Doc2Vec model from path and provides vectorization function."""
    model_name: str
    model_train_data: str
    doc2vec_data_frac: float
    doc2vec_epochs: int
    doc2vec_min_count: int
    model_path: str
    model_type: str = "doc2vec"

    model: Any

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # load model
        print(f"Load Doc2Vec model - {self.model_name}...")
        self.model = Doc2Vec.load(self.model_path)


    def vectorize(self, X, data_col=None):
        if data_col is None:
            data_col = self.text_col
        X = X[data_col]

        # text lowered and split into list of tokens
        print("Pre-process data...")
        X = X.progress_apply(lambda x: simple_preprocess(x))

        # infer vectors from model
        print("Infer doc vectors...")
        docvecs = X.progress_apply(lambda x: self.model.infer_vector(x))
        return list(docvecs)


class WordEmbeddingPooling(EmbeddingModel):
    """Holds word embedding pool model from flair and provides vectorizaiton function."""

    model_type: str = "wordembeddingpool"

    @staticmethod
    def embed(x, model, dim):
        try:
            model.embed(x)
            return x.embedding.detach().cpu().numpy()
        except RuntimeError:
            return np.zeros(dim)

    def vectorize(self, X, data_col=None):
        if data_col is None:
            data_col = self.text_col
        X = X[data_col]

        # init embedding model
        print(f"Load {self.model_name} model ...")
        w_emb = WordEmbeddings(self.model_name)
        model = DocumentPoolEmbeddings([w_emb], fine_tune_mode='nonlinear')

        # convert to Sentence objects
        print("Convert to Sentence objects ...")
        X = X.str.lower()
        sentences = X.progress_apply(lambda x: Sentence(x))

        # get vectors from BERT
        print(f"Get {self.model_name} embeddings ...")
        docvecs = sentences.progress_apply(lambda x: self.embed(
            x, model, model.embedding_flex.out_features))
        docvecs = np.vstack(docvecs)
        return list(docvecs)



class RNNEmbedding(EmbeddingModel):
    """Holds RNN model from flair and provides vectorizaiton function."""
    model_path: str
    model_type: str = "grnn"

    @staticmethod
    def embed(x, model, dim):
        try:
            model.embed(x)
            return x.get_embedding().detach().cpu().numpy()
        except RuntimeError:
            return np.zeros(dim)

    def vectorize(self, X, data_col=None):
        if data_col is None:
            data_col = self.text_col
        X = X[data_col]

        # init embedding model
        print(f"Load {self.model_name} model ...")
        classifier = TextClassifier.load(self.model_path)
        model = classifier.document_embeddings

        # convert to Sentence objects
        print("Convert to Sentence objects ...")
        X = X.str.lower()
        sentences = X.progress_apply(lambda x: Sentence(x))

        # get vectors from BERT
        print(f"Get {self.model_name} embeddings ...")
        docvecs = sentences.progress_apply(lambda x: self.embed(
            x, model, classifier.document_embeddings.embedding_length))
        docvecs = np.vstack(docvecs)
        return list(docvecs)



class TransformerModel(EmbeddingModel):
    """Holds Transformer model from flair (Huggingface) and provides vectorizaiton function."""
    model_size_params: int
    model_type: str = "transformer"

    def vectorize(self, X, data_col=None):
        if data_col is None:
            data_col = self.text_col
        X = X[data_col]

        # init embedding model
        print(f"Load {self.model_name} model ...")
        model = TransformerDocumentEmbeddings(self.model_name, fine_tune=False)

        # convert to Sentence objects
        print("Convert to Sentence objects ...")
        X = X.str.lower()
        sentences = X.progress_apply(lambda x: Sentence(x))

        # get vectors from BERT
        print(f"Get {self.model_name} embeddings ...")
        docvecs = sentences.progress_apply(lambda x: model.embed(x))
        docvecs = sentences.progress_apply(lambda x: x.embedding.cpu().numpy())
        return list(docvecs)


class PreComputed(EmbeddingModel):
    model_name: str
    model_train_data: str
    """Input is the column of precomputed vectors which is simply returned unchanged."""
    def vectorize(self, X, data_col=None):
        if data_col is None:
            data_col = self.text_col
        return X[data_col]
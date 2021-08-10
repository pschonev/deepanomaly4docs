from abc import ABC, abstractmethod
from typing import List, Any, Dict
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN, all_points_membership_vectors
from pydantic.main import BaseModel
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from ivis import Ivis
from sklearn.decomposition import PCA as PCAR
from sklearn.manifold import TSNE
from umap import UMAP
from src.utils import next_path, product_dict, get_scores, sample_data
from pydantic import BaseModel, Field


class TestData(BaseModel):
    """Loads from path, samples and holds outlier dataset."""
    path: str
    name: str
    fraction: List[float]
    contamination: List[float]
    seed: List[int]
    min_len: int = 5
    df: Any

    def load_data(self):
        print(f"Loading data from {self.path} to DataFrame...")
        self.df = pd.read_pickle(self.path)
        return self

    def remove_short_texts(self):
        n_before = self.df.shape[0]
        self.df = self.df[self.df['text'].map(len) > self.min_len]
        print(
            f"Removed {n_before - self.df.shape[0]} rows with doc length below {self.min_len}.")
        return self

    def cartesian_params(self):
        return list(product_dict(fraction=self.fraction, contamination=self.contamination, seed=self.seed))

    def sample_data(self, fraction, contamination, seed):
        return sample_data(self.df, fraction, contamination, seed)


class DimensionReducer(BaseModel, ABC):
    """Abstract class to reduce dimensions of document vectors."""
    @abstractmethod
    def reduce_dims(self, docvecs):
        pass


class NoReduction(DimensionReducer):
    """No reduction on document vectors."""
    dim_red_name: str = "NoRed"
    def cartesian_params(self):
        return [dict()]

    def reduce_dims(self, docvecs):
        return docvecs



class SklearnReducer(DimensionReducer):
    """Holds dimensionality reduction model that complies to scikit-learn syntax."""
    dim_red_name: str
    dim_reducer: str
    as_numpy: bool
    kwargs: dict = Field(default_factory=dict)

    algos: dict = Field(default_factory=lambda: {"umap": UMAP})

    def cartesian_params(self):
        return product_dict(**self.kwargs)

    def reduce_dims(self, docvecs, **kwargs):
        if self.as_numpy:
            try:
                self.dim_reducer = self.algos[self.dim_reducer.lower()]
            except AttributeError:
                pass
            docvecs = np.array(docvecs)
            return self.dim_reducer(**kwargs).fit_transform(docvecs)



class OutlierDetector(BaseModel, ABC):
    """Abstract class to detect outliers."""
    @abstractmethod
    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, **kwargs):
        pass



class PyodDetector(OutlierDetector):
    """Holds outlier detector model from PyOD library and provides outlier detection function."""
    pyod_model: str
    outlier_detector: str
    kwargs: dict = Field(default_factory=dict)

    algos: dict = Field(default_factory=lambda: 
    {"autoencoder": AutoEncoder})

    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, **kwargs):
        print(f"Outlier detection using pyod's {self.pyod_model}")
        try:
            self.pyod_model = self.algos[self.pyod_model.lower()]
        except AttributeError:
            pass
        od = self.pyod_model(**kwargs)
        od.fit(dim_reduced_vecs)

        out_pred = od.labels_
        out_pred[out_pred == 1] = -1
        out_pred[out_pred == 0] = 1

        scores = get_scores(outlier_labels, out_pred, scores=scores)
        scores.update(**kwargs)
        out_f1 = scores["out_f1"]
        print(f"{kwargs}\nOut_f1: {out_f1}\n\n")
        return scores, out_pred

    def cartesian_params(self):
        return product_dict(**self.kwargs)



class DimRedOutlierDetector(OutlierDetector):
    """Holds dimensionality reduction algorithm and provides function to threshold a range to detect outliers on one-dimensional output."""
    dem_red_outlier_model: Any
    outlier_detector: str
    as_numpy: bool
    kwargs: dict = Field(default_factory=dict)

    @staticmethod
    def reject_outliers(sr, iq_range=0.5):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = np.quantile(sr, [pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        return ((np.abs(sr - median)) >= iqr/2)

    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, **kwargs):
        od = self.dem_red_outlier_model(**kwargs)
        if self.as_numpy:
            dim_reduced_vecs = np.array(dim_reduced_vecs)
        preds = od.fit_transform(dim_reduced_vecs)
        preds = preds.astype(float)

        preds = self.reject_outliers(preds, iq_range=1.0-contamination)
        preds = [-1 if x else 1 for x in preds]

        scores = get_scores(outlier_labels, preds, scores=scores)
        scores.update(**kwargs)
        out_f1 = scores["out_f1"]
        print(f"{kwargs}\nOut_f1: {out_f1}\n\n")
        return scores, preds

    def cartesian_params(self):
        return product_dict(**self.kwargs)



class HDBSCAN_GLOSH(OutlierDetector):
    """Provides function to detect outliers with HDBSCAN's GLOSH."""
    min_cluster_size: List[int]
    allow_noise: List[bool]
    outlier_detector: str = "HDBSCAN_GLOSH"

    def predict(self, dim_reduced_vecs, outlier_labels, scores, contamination, min_cluster_size, allow_noise):
        print("Clustering ...")
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            prediction_data=True, metric="euclidean").fit(dim_reduced_vecs)
        print("Get prediction data ...")
        clusterer.generate_prediction_data()

        try:
            cluster_pred = clusterer.labels_ if allow_noise else np.argmax(
                all_points_membership_vectors(clusterer)[:, 1:], axis=1)
        except IndexError:
            print(
                "Got IndexError and will not enforce cluster membership (allow noise) ...")
            print(all_points_membership_vectors(clusterer))
            cluster_pred = clusterer.labels_

        # scoring
        print("Get scores ...")

        # GLOSH
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
        outlier_pred = np.where(
            clusterer.outlier_scores_ > threshold, -1, 1)

        scores["cluster_n"] = len(np.unique(clusterer.labels_))
        scores["homogeneity"] = homogeneity_score(
            outlier_labels, cluster_pred)
        scores["completeness"] = completeness_score(
            outlier_labels, cluster_pred)
        scores["v_measure"] = v_measure_score(outlier_labels, cluster_pred)

        scores = get_scores(outlier_labels, outlier_pred, scores=scores)

        print(f"Homogeneity - {homogeneity_score(outlier_labels, cluster_pred)*100:.1f}  \
                cluster_n - {len(np.unique(clusterer.labels_))}")

        return scores, clusterer.outlier_scores_

    def cartesian_params(self):
        return product_dict(min_cluster_size=self.min_cluster_size, allow_noise=self.allow_noise)



class EvalRun(BaseModel):
    """Holds objects for data, embedding model, dimensionality reduction and outlier detection. Also handles result path and iteration counting."""
    name: str
    emb_model: Dict[str, EmbeddingModel]
    data: Dict[str, TestData]
    dim_red: Dict[str, DimensionReducer]
    od_det: Dict[str, OutlierDetector]
    res_folder: str = ""
    res_path: str = ""
    total_iter: int = 0
    current_iter: int = 1

    def init_iter_counter(self):
        # convert to lists
        self.emb_model = list(self.emb_model.values())
        self.data = list(self.data.values())
        self.dim_red = list(self.dim_red.values())
        self.od_det = list(self.od_det.values())

        model_perm = len(self.emb_model)
        test_data_perm = sum(len(x.cartesian_params())
                             for x in self.data)
        dim_red_perm = sum(len(x.cartesian_params())
                           for x in self.dim_red)
        out_detect_perm = sum(len(x.cartesian_params())
                              for x in self.od_det)
        self.total_iter = model_perm*test_data_perm*dim_red_perm*out_detect_perm

        print(f"Evaluating {self.total_iter} parameter permutations.")
        return self

    def init_result_path(self):
        self.res_path = next_path(
            self.res_folder + "%04d_" + self.name + ".tsv")
        print(f"Saving results to {self.res_path}")
        return self

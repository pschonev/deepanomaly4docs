
import umap
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin



class OutlierDetector(BaseEstimator, TransformerMixin):

    def __init__(self,
                 data_path = "/home/philipp/projects/dad4td/dad4tdenv/data/processed",
                 text="text",
                 labels="topic",
                 embedder=TfidfVectorizer(min_df=5, stop_words='english'),
                 dim_reducer = umap.UMAP(metric='hellinger'),
                 dim=2,
                 graph=
                 ):

        self.base_path = data_path
        self.text = text
        self.labels = labels
        self.embedder = embedder
        self.dim_reducer = dim_reducer
        self.dim = dim


pipe = Pipeline
""" 
Helper Classes
"""

from river import stream
from river import cluster


class Dimensionality:
    """Class we swap in instead of calling UMAP"""

    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings

    def fit(self, X):
        return self

    def transform(self, X):
        return self.reduced_embeddings


class River:
    """Wrapper Class for online topic models, allows partial fitting if documents arrive in streams."""
    def __init__(self, model):
        self.model = model
    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)
        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)
        self.labels_ = labels
        return self

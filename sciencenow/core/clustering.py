from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import warnings

from pandas import (
    DataFrame,
    read_feather,
    concat as concat_dataframe,
)

from numpy import (
    load as load_array,
    save as save_array,
    concatenate as concat_array,
    array,
)

from river import stream


class River:
    """
    Clustering Model for Online Topic Modeling.
    Allows partial fitting if documents arrive in streams.
    """
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
    

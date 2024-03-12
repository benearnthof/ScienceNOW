from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

from numpy import (
    load as load_array,
    save as save_array,
    ndarray
)

from sciencenow.config import (
    setup_params,
)

# Make version of UMAP dependent to config so we can choose the correct version depending on the deployment 
if setup_params["GPU"]:
    from cuml.manifold import UMAP # need the GPU implementation to process 2 million embeddings
else:
    from umap import UMAP


class Reducer(ABC):
    """
    Abstract Base Class for all Dimensionality Reduction functionality.
    """

    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def reduce(self):
        raise NotImplementedError
    

class UmapReducer(Reducer):
    """
    UMAP extension of Reducer class.

    Args: 
        neighbors: int that specifies number of neighbors to consider in UMAP
        components: int that specifies target dimensionality
        metric: a metric compatible with UMAP. Default: Cosine for text embeddings.
        source: Optional string that specifies location of preprocessed embeddings.
        target: Optional string that specifies where to write embeddings to disk.
        data: ndarray of embeddings that will be reduced to dimension of `components`.
        labels: ndarray of numeric labels used for supervised UMAP.
    """
    def __init__(
            self,
            setup_params: Dict[str, Any],
            source: Optional[str]=None,
            target: Optional[str]=None,
            data: ndarray=None,
            labels: ndarray=None,
            ) -> None:
        super().__init__()
        self.umap_model = UMAP(
            setup_params["umap_neighbors"], 
            setup_params["umap_components"], 
            setup_params["umap_metric"]
        )
        self.source=Path(source) if source is not None else None
        self.target=Path(target) if source is not None else None
        self.data=data
        self.labels=labels
        self.reduced_embeddings=None

    def load(self) -> None: 
        """
        Loads Embeddings that have been previously saved with `self.save`
        """
        if not self.source.exists():
            raise FileNotFoundError(f"No precomputed embeddings found in {self.source}. Call `reduce` first.")
        self.reduced_embeddings = load_array(self.source)
        print(f"Successfully loaded {self.reduced_embeddings.shape[0]} reduced embeddings")

    def save(self) -> None:
        if self.target is None:
            raise NotImplementedError("Unable to save embeddings, no target provided.")
        if self.embeddings is None:
            raise NotImplementedError("Cannot save empty array, call `embed` first.")
        save_array(self.target, self.reduced_embeddings, allow_pickle=False)
        print(f"Successfully saved {self.reduced_embeddings.shape[0]} reduced embeddings to disk.")
    
    def reduce(self) -> None:
        """
        Obtain Reduced Embeddings with UMAP to save time in Topic Modeling steps.
        Will be called after all preprocessing steps like filtering sampling and merging have been completed.
        This drastically speeds up computation and allows us to deploy on CPU only instances.
        """
        if self.data is None:
            raise NotImplementedError("Cannot reduce empty data, aborting procedure.")
        self.reduced_embeddings = self.umap_model.fit_transform(self.data, y=self.labels)
        print(f"Successfully reduced embeddings of subset from {self.data.shape} to {self.reduced_embeddings.shape}")


class Dimensionality:
    """
    Auxiliary class that is used to directly pass precomputed reduced embeddings to BERTopic.
    Not used anymore since we need to recompute reduced embeddings every time the dataset changes.
    """

    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings

    def fit(self, X):
        return self

    def transform(self, X):
        return self.reduced_embeddings

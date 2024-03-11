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

from sciencenow.core.dimensionality import Reducer


class TopicModel(ABC):
    """
    Abstract Base Class for Topic Models.
    """
    data: DataFrame
    embeddings: array
    reducer: Reducer
    corpus: str
    vocab: str

    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        raise NotImplementedError

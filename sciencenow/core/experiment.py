from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

from pandas import DataFrame
from numpy import array
from tqdm import tqdm

from sciencenow.core.pipelines import Pipeline
from sciencenow.core.dataset import Dataset, DatasetMerger
from sciencenow.core.model import TopicModel
from sciencenow.core.dimensionality import Reducer


class Experiment(ABC):
    """
    Abstract Base Class for Experiments.
    Unifies Loading, Preprocessing, Merging, Model Setup, Training, and Evaluation.

    Args: 
        model: TopicModel that will be trained and evaluated.
        postprocessing: List of postprocessing pipelines to be executed when training is complete.
        
    """
    model: TopicModel
    postprocessing: List[Pipeline]

    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def run(self):
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        raise NotImplementedError


class BERTopicExperiment(Experiment):
    """
    Extends Experiment to setup, train, and evaluate a BERTopic model.
    """
    def __init__(
            self,
            model: TopicModel,
            postprocessing: List[Pipeline],
            setup_params: Dict[str, Any]
            ) -> None:
        """
        Initialize Experiment object.

        Args: 
            merger: DatasetMerger that contains data & document embeddings that will be used for model
            reducer: Reducer that specifies how document embeddings will be reduced for clustering
            setup_params: Dict that provides auxiliary parameters.
        """
        super().__init__()
        self.model = model
        self.postprocessing = postprocessing
        self.setup_params = setup_params
    
    def load(self):
        pass

    def save(self):
        pass

    def run(self):
        print(f"Training topic model with params: {self.setup_params}")
        self.model.train()

    def eval(self):
        self.postprocessing.execute(input=self.model)

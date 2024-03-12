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
from sciencenow.core.dimensionality import UmapReducer


class Experiment(ABC):
    """
    Abstract Base Class for Experiments.
    Unifies Loading, Preprocessing, Merging, Model Setup, Training, and Evaluation.

    Args: 
        model: TopicModel that will be trained and evaluated.
        postprocessing: List of postprocessing pipelines to be executed when training is complete.
        merger: DatasetMerger that contains data & document embeddings that w
    """
    model: TopicModel
    postprocessing: List[Pipeline]
    merger: DatasetMerger

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
            datapath: List[str],
            pipeline: List[Pipeline],
            model: TopicModel,
            postprocessing: List[Pipeline],
            setup_params: Dict[str, Any]
            ) -> None:
        super().__init__()
        self.datapath = datapath
        self.pipeline = pipeline
        self.model = model
        self.postprocessing = postprocessing
        self.setup_params = setup_params
        self.reducer = None
    
    def load(self):
        pass

    def save(self):
        pass

    def setup(self):
        #### Caching can be done by setting up a separate pipeline that takes in a df directly from Experiment
        


    def run(self):
        pass
    
    def eval(self):
        pass

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from pandas import DataFrame, read_feather
import warnings

from sciencenow.core.pipelines import Pipeline

class Dataset(ABC):
    """
    Abstract Base Class for all Datasets
    """
    path: str
    pipeline: Pipeline
    source: Any
    data: DataFrame

    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    def preprocess(self):
        self.pipeline.execute()


class PubmedDataset(Dataset):
    """
    Base Class for Pubmed data saved as a .txt file.
    """
    def __init__(self, path:str, pipeline: Pipeline) -> None:
        super().__init__()
        self.path = Path(path)
        self.pipeline = pipeline
        self.source = None
        self.data = None
        self.taxonomy = None

    def save(self, path):
        """
        Method to store a preprocessed dataframe as .feather file.
        For much faster writing and loading to and from disk.
        """
        if isinstance(self.data, DataFrame):
            self.data.to_feather(Path(path))
            print(f"Stored dataframe at {path}.")
        else:
            warnings.warn("No DataFrame found. Make execute preprocessing pipeline first.")

    def load(self, path):
        """
        Method to load a preprocessed dataframe stored as .feather format.
        """
        filepath = Path(path)
        if filepath.exists():
            if not filepath.suffix == ".feather":
                raise NotImplementedError(f"Data must be stored in .feather format, found {filepath.suffix}")
            self.data = read_feather(filepath)
            print(f"Loaded dataframe from {path}")



class ArxivDataset(PubmedDataset):
    """
    Base Class for Arxiv data saved in a .json snapshot as provided by the OAI.
    https://github.com/mattbierbaum/arxiv-public-datasets
    """
    def __init__(self, path:str, pipeline:Pipeline) -> None:
        super().__init__(path, pipeline)
        self.path=path
        self.pipe=pipeline

    def load_taxonomy(self, path: str) -> None:
        """
        Loads Arxiv Taxonomy for semisupervised models.
        Will load the Arxiv Category Taxonomy (https://arxiv.org/category_taxonomy) as a dictionary from disk.
        This provides us with a map from category labels to plaintext and is also used to obtain numeric class labels
        for the semisupervised model. 
        """
        if not Path(path).exists():
            raise NotImplementedError(f"No taxonomy found in {path}.")
        with open(path, "r") as file:
            taxonomy = [line.rstrip() for line in file]
        taxonomy = [line.split(":") for line in taxonomy]
        taxonomy_dict = {line[0]: line[1].lstrip() for line in taxonomy}
        # add missing label to taxonomy
        taxonomy_dict["cs.LG"] = "Machine Learning"
        self.taxonomy = taxonomy_dict
        print(f"Successfully loaded {len(self.taxonomy)} labels from {path}.")

    
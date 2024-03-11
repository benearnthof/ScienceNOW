from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import warnings

from pandas import DataFrame, read_feather
from numpy import array

from sciencenow.core.pipelines import Pipeline
from sciencenow.core.embedding import ArxivEmbedder, Embedder

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
        self.taxonomy = None
        self.data = None

    def load_taxonomy(self, path: str) -> None:
        """
        Loads Arxiv Taxonomy for semisupervised models.
        Will load the Arxiv Category Taxonomy (https://arxiv.org/category_taxonomy) as a dictionary from disk.
        This provides us with a map from category labels to plaintext and is also used to obtain numeric class labels
        for the semisupervised model. 
        """
        if path is None:
            warnings.warn("No path specified, returning None.")
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
    
    def execute_pipeline(self, input):
        self.data = self.pipeline.execute(input)



# Every Dataset is defined as the result of the execution of a series of preprocessing steps => Pipeline
# The index of every dataset corresponds to the ID of the embedding vector saved to disk
# For model preprocessing we need only to load the partially reduced embeddings for the entire dataset
        # Dimension (2.3 million, 100/200/500/768)
# Select the corresponding embedding vectors by their indices
# And then pass them to the reducer object to perform UMAP for the small subset we're interested in.
        
class DatasetMerger(ABC):
    """
    Abstract Base Class for merging two datasets.
    The central problem classes that inherit from this Class adress is the unification of two distinct 
    datasets that may have different sources or be from the same source. 
    Every dataset has a collection of documents, defined as "Abstracts". 
    Every document has a corresponding SentenceEmbedding. 
    We need a merger class so we can pick the correct SentenceEmbeddings, and avoid having to recompute
    the embeddings every time we train a Topic model. This saves anywhere from a couple of minutes of 
    computation for small datasets, to about 80 minutes on an A100 GPU for the full list 
    of 2.3 Million Arxiv Abstracts.

    Args: 
        source: some Dataset we wish to combine with target
        target: some Dataset we wish to combine with source
        source_embedder: Embedder class that stores the embeddings of the source Dataset
        target_embedder: Embedder class that stores the embeddings of the target Dataset
        data: Dataframe that stores result of the merging operation
        embeddings: array that stores the combined embeddings resulting from the merging operation
    """
    source: Dataset
    target: Dataset
    source_embedder: Embedder
    target_embedder: Embedder
    data: DataFrame
    embeddings: array

    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def merge(self):
        raise NotImplementedError
    

class BasicMerger(DatasetMerger):
    """
    Basic Merger class that just combines the source and target Datasets by concatenation
    """
    def __init__(self) -> None:
        super().__init__()

    def load(self, path):
        pass

    def save(self, path):
        pass

    def merge(self)
        pass

from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path
import warnings
from string import Template
from collections import Counter
from random import (
    shuffle,
    choices,
)

from pandas import (
    DataFrame,
    read_feather,
    concat as concat_dataframe,
)

from numpy import (
    load as load_array,
    save as save_array,
    rint as random_int,
    min as array_min,
    concatenate as concat_array,
    any as any_array,
    array_split,
    array,
    in1d,
)

from sciencenow.core.pipelines import Pipeline
from sciencenow.core.embedding import ArxivEmbedder, Embedder

class Dataset(ABC):
    """
    Abstract Base Class for all Datasets
    """
    path: str
    pipeline: Pipeline
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
        if not filepath.exists():
            raise FileNotFoundError(f"No file found at specified location: {filepath}")
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

    # load and save allow us to do caching in case we already performed setup
    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def merge(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_id(self):
        raise NotImplementedError
    

class BasicMerger(DatasetMerger):
    """
    Basic Merger class that just combines the source and target Datasets by concatenation
    """
    def __init__(
            self,
            source: Dataset,
            target: Dataset,
            source_embedder: Embedder,
            target_embedder: Embedder,
            ) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.source_embedder = source_embedder
        self.target_embedder = target_embedder
        self.data = None
        self.embeddings = None

    def load(self, datapath: str, embedpath: str) -> None:
        """
        Load merged dataset from disk.

        Args:
            datapath: string that specifies where merged data is located on disk.
            embedpath: string that specifies where respective embeddings are located on disk.
        """
        if not datapath:
            raise NotImplementedError(f"datapath must be provided to load from disk.")
        if not embedpath: 
            raise NotImplementedError(f"embedpath must be provided to load from disk")
        datapath, embedpath = Path(datapath), Path(embedpath)
        
        if not datapath.exists():
            raise FileNotFoundError(f"No file found at specified datapath: {datapath}")
        if not embedpath.exists():
            raise FileNotFoundError(f"No file found at specified embedpath: {embedpath}")
        if not datapath.suffix == ".feather":
            raise NotImplementedError(f"Data must be stored in .feather format. Found {datapath.suffix}")
        if not embedpath.suffix == ".npy":
            raise NotImplementedError(f"Embeddings must be stored in .npy format. Found {embedpath.suffix}")
        self.data = read_feather(datapath)
        self.embeddings = load_array(embedpath)
        print(f"Loaded data and embeddings from {datapath} & {embedpath}")

    def save(self, datapath:str, embedpath: str) -> None:
        """
        Save merged dataset to disk.
        Args:
            datapath: string that specifies where merged data will be stored.
            embedpath: string that specifies where respective embeddings will be stored.
        """
        if not datapath:
            raise NotImplementedError(f"datapath must be provided to load from disk.")
        if not embedpath: 
            raise NotImplementedError(f"embedpath must be provided to load from disk")
        datapath, embedpath = Path(datapath), Path(embedpath)

        if isinstance(self.data, DataFrame) and isinstance(self.embeddings, array):
            self.data.to_feather(datapath)
            save_array(embedpath, self.embeddings, allow_pickle=False)
            print(f"Stored dataframe and embeddings at {datapath}, {embedpath}.")
        else:
            raise NotImplementedError("Data or embeddings missing, aborting save.")

    def get_id(self, setup_params: Dict[str, Any], primary: bool=True) -> str:
        """
        Returns a unique ID for a set of given setup parameter
        # TODO: Add to load and save functions to automate caching of datasets
        """
        if not primary:
            template = Template("$secondary_target $secondary_startdate $secondary_enddate $secondary_proportion $trend_deviation $n_trends")
        else: 
            template = Template("$target $startdate $enddate $threshold $limit")
        return template.substitute(**setup_params)


    def merge(self) -> None:
        """
        Basic merge that picks embeddings based on the indices present in Datasets and then 
        concatenates the respective data and embeddings.
        Idea: 
            Both Datasets have already been preprocessed to the needed proportions. 
            All that remains is the concatenation of both of them.
        """
        if not isinstance(self.source, Dataset) or not isinstance(self.target, Dataset):
            raise NotImplementedError("Please provide both source and target Dataset.")
        if not isinstance(self.source_embedder, Embedder) or not isinstance(self.target_embedder, Embedder):
            raise NotImplementedError("Please provide both source and target Embedder.")
        
        if self.source_embedder.embeddings is None or self.target_embedder.embeddings is None:
            raise NotImplementedError("Please provide both source and target embeddings.")
        
        source_ids = self.source.data.index.tolist()
        target_ids = self.target.data.index.tolist()
        source_embeddings = self.source_embedder.embeddings[source_ids]
        target_embeddings = self.target_embedder.embeddings[target_ids]
        self.data = concat_dataframe([self.source.data, self.target.data])
        self.embeddings = concat_array((source_embeddings, target_embeddings))
        # Merger Object merges data and retrieves embeddings
        # Then we need to obtain updated numeric labels
        # Then we can pass the merged embeddings to the UmapReducer class

class SyntheticMerger(BasicMerger):
    """
    Extends BasicMerger to be able to handle 
    
    Args:
        source: Dataset trends will be sampled from
        target: Dataset trends will be added to
        source_embedder: Embedder providing sentence embeddings for source data
        target_embedder: Embedder providing sentence embeddings for target data
    """
    def __init__(
            self, 
            source: Dataset, 
            target: Dataset, 
            source_embedder: Embedder, 
            target_embedder: Embedder
            ) -> None:
        super().__init__(source, target, source_embedder, target_embedder)
        
        self.papers_per_bin = None

    def merge(self, setup_params: Dict[str, Any]) -> None:
        """
        Overwrites basic merge in favor of a custom sampling procedure based on setup parameters.
        Assumes both source and target dataset already have numeric labels & plaintext labels.
        At least target dataset must have column "v1_datetime".
        """
        target, source = self.target.data, self.source.data
        # synthetic trend will be sampled from source and added to target
        # simplest idea: Insert proportion of source into target
        amount = int(len(target) * setup_params["secondary_proportion"])
        # we know by setup through steps that the source dataset is large enough to sample from
        synthetic_set = source.sample(n = amount)
        print(f"Selected {amount} papers to be merged with subset.")
        synthetic_set = self.adjust_timestamps(synthetic_set, setup_params)
        # obtain merged embeddings
        synthetic_ids = synthetic_set.index.tolist()
        target_ids = target.index.tolist()
        synthetic_embeddings = self.source_embedder.embeddings[synthetic_ids]
        target_embeddings = self.target_embedder.embeddings[target_ids]
        # adjust numeric labels to remove potential class overlaps for supervised models
        synthetic_set = self.adjust_labels(target, synthetic_set)
        self.data = concat_dataframe([target, synthetic_set])
        self.embeddings = concat_array((target_embeddings, synthetic_embeddings)) # TODO: Order should be preserved, need to write test

    def adjust_timestamps(self, synthetic_set: DataFrame, setup_params: Dict[str, Any]) -> DataFrame:
        """
        Adjusts timestamps of source data to simulate a trend in target data upon merging.
        Args:
            setup_params: Dict that specifies how trend will be introduced to the data.
        """
        target_timestamps = self.target.v1_datetime.tolist()
        bins, n_trends, deviation = setup_params["nr_bins"], setup_params["n_trends"], setup_params["trend_deviation"]
        trend_multipliers = [1] * (bins - n_trends) + [deviation] * n_trends
        shuffle(trend_multipliers)

        # normalize so we can calculate how many papers should fall into each bin
        trend_multipliers = array([float(i) / sum(trend_multipliers) for i in trend_multipliers])
        self.papers_per_bin = random_int(self.trend_multipliers * len(synthetic_set)).astype(int)
        print(f"PAPERS PER BIN: {self.papers_per_bin}")
        
        # make sure that samples will match in length
        new_set = synthetic_set[0:sum(self.papers_per_bin)-1]
        new_timestamps = []
        binned_timestamps = array_split(target_timestamps, bins)

        for i, n in enumerate(self.papers_per_bin):
            sample = choices(binned_timestamps[i].tolist(), k=n)
            new_timestamps.extend(sample)
            
        new_timestamps = new_timestamps[0:len(new_set)]
        assert len(new_timestamps) == len(new_set)
        new_set.v1_datetime = new_timestamps
        return new_set
    
    def adjust_labels(self, target, synthetic):
        """
        Method to adjust Labels.
        General Problem: sciencenow.core.steps.ArxivGetNumericLabelsStep only cares about the 
        number of distinct plaintext labels present in target and synthetic separately.
        We need to adjust the labels of the synthetic data to avoid label overlaps.
        """
        target_labels = array(target.numeric_labels.tolist())
        synthetic_labels = array(synthetic.numeric_labels.tolist())
        adjusted_synthetic = synthetic_labels.copy()
        while any_array(in1d(adjusted_synthetic, target_labels)):
            # Find the minimum element in the overlap
            overlap_min = array_min(adjusted_synthetic[in1d(adjusted_synthetic, target_labels)])
            # Add a constant value to all elements in synthetic greater than or equal to overlap_min
            adjusted_synthetic[adjusted_synthetic >= overlap_min] += 1
        
        synthetic = synthetic.assign(numeric_labels=adjusted_synthetic)
        return synthetic


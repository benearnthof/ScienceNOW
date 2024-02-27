"""
Enumerators to handle loading of config dynamically.
"""
from pathlib import Path
from enum import Enum
from omegaconf import OmegaConf
import os

fpath = os.path.realpath(__file__)
fpath = Path(fpath)

CONFIGPATH = fpath.parent / "secrets.yaml"
assert CONFIGPATH.exists()
config = OmegaConf.load(CONFIGPATH)

class FP(Enum):
    """
    All File Paths we need to supply to ArxivProcessor
    """
    SNAPSHOT = Path(config.ARXIV_SNAPSHOT)
    EMBEDS = Path(config.EMBEDDINGS)
    REDUCED_EMBEDS = Path(config.REDUCED_EMBEDDINGS)
    FEATHER = Path(config.FEATHER_PATH)
    TAXONOMY = Path(config.TAXONOMY_PATH)
    VOCAB = Path(config.VOCAB_PATH)
    CORPUS = Path(config.CORPUS_PATH)

class PARAMS(Enum):
    """
    All Fixed Model Params we need to instantiate ArxivProcessor
    """
    SENTENCE_MODEL = config.SENTENCE_MODEL
    UMAP_NEIGHBORS = config.UMAP_NEIGHBORS
    UMAP_COMPONENTS = config.UMAP_COMPONENTS
    UMAP_METRIC = config.UMAP_METRIC


class TM_PARAMS(Enum):
    """
    Wrapper class for Topic Model hyper parameters.
    Params:
        VOCAB_THRESHOLD:`int` that specifies how many times a word must occur in the corpus for 
            it to be contained in the topic model vocabulary. (This is separate from the vocab used for evaluation.)
    """
    VOCAB_THRESHOLD=config.VOCAB_THRESHOLD
    TM_VOCAB_PATH=config.TM_VOCAB_PATH
    TM_TARGET_ROOT=config.TM_TARGET_ROOT
    EVAL_ROOT=config.EVAL_ROOT

setup_params = {
    "samples": 1, # hdbscan samples
    "cluster_size": 25, # hdbscan minimum cluster size
    "startdate": "01 01 2020", # if no date range should be selected set startdate to `None`
    "enddate": "31 01 2020",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "secondary_target": None, # for synthetic trend extraction
    "secondary_startdate": "01 01 2020",
    "secondary_enddate": "31 12 2020",
    "secondary_proportion": 0.1,
    "trend_deviation": 1.5, # value between 1 and 2 that determines how much more papers will be in the "trending bins"
                            # compared to the nontrending bins
    "n_trends": 1,
    "threshold": 0,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": 0,
    "recompute": False,
    "nr_topics": None,
    "nr_bins": 4, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": 4, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": False, # For dynamic model
    "global_tuning": False, # For dynamic model
    "limit": None,
    "subset_cache": "C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\subset_cache",
}

online_params = {# For online DBSTREAM https://riverml.xyz/latest/api/cluster/DBSTREAM/
    "clustering_threshold": 1.0, # radius around cluster center that represents a cluster
    "fading_factor": 0.01, # parameter that controls importance of historical data to current cluster > 0
    "cleanup_interval": 2, # time interval between twwo consecutive time periods when the cleanup process is conducted
    "intersection_factor": 0.3, # area of the overlap of the micro clusters relative to the area cover by micro clusters
    "minimum_weight": 1.0 # minimum weight for a cluster to be considered not "noisy" 
}

# config where values that should not be displayed by Gradio app are set to None
gradio_setup_params = {
    "samples": None, # hdbscan samples
    "cluster_size": 25, # hdbscan minimum cluster size
    "startdate": "01 01 2020", # if no date range should be selected set startdate to `None`
    "enddate": "31 01 2020",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "secondary_target": "q-bio", # for synthetic trend extraction
    "secondary_startdate": "01 01 2020",
    "secondary_enddate": "31 12 2020",
    "secondary_proportion": 0.1,
    "trend_deviation": 1.5, # value between 1 and 2 that determines how much more papers will be in the "trending bins"
                            # compared to the nontrending bins
    "n_trends": 1,
    "threshold": None,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": None,
    "recompute": False,
    "nr_topics": None,
    "nr_bins": 4, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": None, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": None, # For dynamic model
    "global_tuning": None, # For dynamic model
    "limit": 5000,
    "subset_cache": None,
}
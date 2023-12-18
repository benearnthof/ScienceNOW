"""
tests for sciencenow.models.train.py
"""

from sciencenow.models.train import ModelWrapper
from sciencenow.utils.wrappers import Dimensionality, River
from cuml.cluster import HDBSCAN

from pandas import Timestamp
from collections import Counter
import numpy as np
import math
from pathlib import Path

#### TEST SETUP
# IF
setup_params = {
    "samples": 1, # hdbscan samples
    "cluster_size": 30, # hdbscan minimum cluster size
    "startdate": "01 01 2020", # if no date range should be selected set startdate to `None`
    "enddate": "31 12 2020",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "secondary_target": None, # for synthetic trend extraction
    "secondary_startdate": "01 01 2020",
    "secondary_enddate": "31 12 2020",
    "secondary_proportion": 0.1,
    "trend_deviation": 1.5, # value between 1 and 2 that determines how much more papers will be in the "trending bins"
                            # compared to the nontrending bins
    "n_trends": 1,
    "threshold": 100,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": 0,
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 52, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": 52, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": False, # For dynamic model
    "global_tuning": False, # For dynamic model
    "limit": None,
    "subset_cache": Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/cache/"),
}

model_type = "dynamic"

# WHEN
wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)

# THEN
assert wrapper.subset.shape == (49380, 18)
assert len(set(wrapper.plaintext_labels)) == 77
assert wrapper.subset.abstract.tolist()[0].startswith("What we discover and see online")
assert wrapper.subset.abstract.tolist()[-1].startswith("In this work, we consider a multi-user mobile edge computing ")
assert wrapper.subset.v1_datetime.tolist()[0] == Timestamp('2020-01-01 00:12:34+0000', tz='UTC')
assert wrapper.subset.v1_datetime.tolist()[-1] == Timestamp('2020-12-31 23:52:28+0000', tz='UTC')
assert wrapper.model_type is not None
assert wrapper.model_type == "dynamic"
assert wrapper.subset_reduced_embeddings is not None
assert wrapper.subset_reduced_embeddings.shape == (wrapper.subset.shape[0], wrapper.processor.PARAMS.UMAP_COMPONENTS.value)
assert wrapper.tm_vocab is None
assert wrapper.topic_model is None
assert wrapper.topics is None
assert wrapper.topic_info is None
assert wrapper.topics_over_time is None

#### Test topic model setup
# IF
# WHEN
wrapper.tm_setup()

# THEN
assert isinstance(wrapper.umap_model, Dimensionality) 
assert wrapper.umap_model.reduced_embeddings.shape == (wrapper.subset.shape[0], wrapper.processor.PARAMS.UMAP_COMPONENTS.value)
assert isinstance(wrapper.cluster_model, HDBSCAN)
assert wrapper.ctfidf_model is not None
assert wrapper.topic_model is not None

#### Test topic model training

# IF
# WHEN
wrapper.tm_train()

# THEN
assert wrapper.topics is not None
assert wrapper.probs is None
assert wrapper.topics_over_time is not None
assert wrapper.topic_info is not None

assert len(wrapper.topics) == wrapper.subset.shape[0]
assert len(set(wrapper.topics)) == 147
assert wrapper.topics_over_time.shape[0] == 5988

#### Test semisupervised model setup
# IF 
setup_params = {
    "samples": 30, # hdbscan samples
    "cluster_size": 30, # hdbscan minimum cluster size
    "startdate": "01 01 2020", # if no date range should be selected set startdate to `None`
    "enddate":"31 12 2020",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "threshold": 100,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": 0.1,
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 52, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": 52, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": False, # For dynamic model
    "global_tuning": False, # For dynamic model
}

model_type = "semisupervised"

# WHEN
wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)

# THEN
assert wrapper.plaintext_labels is not None
assert wrapper.numeric_labels is not None
assert len(set(wrapper.numeric_labels)) == 78
assert math.isclose(Counter(wrapper.numeric_labels)[-1] / len(wrapper.numeric_labels) , 0.1, rel_tol = 0.02)

#### Test semisupervised model training

# IF 
# WHEN
wrapper.tm_setup()
wrapper.tm_train()

# THEN
assert wrapper.topics is not None
assert wrapper.probs is not None
assert wrapper.topics_over_time is not None
assert wrapper.topic_info is not None

assert len(wrapper.topics) == wrapper.subset.shape[0]
assert len(wrapper.probs) == wrapper.subset.shape[0]
assert len(set(wrapper.topics)) == 139
assert wrapper.topics_over_time.shape[0] == 5610


#### Test online model setup
# IF 
setup_params = {
    "samples": 30, # hdbscan samples
    "cluster_size": 30, # hdbscan minimum cluster size
    "startdate": "01 01 2020", # if no date range should be selected set startdate to `None`
    "enddate":"31 01 2020",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "threshold": 100,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": 0.1,
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 4, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": 4, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": False, # For dynamic model
    "global_tuning": False, # For dynamic model
}

model_type = "online"

# WHEN
wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)

# THEN
assert wrapper.plaintext_labels is not None
assert wrapper.numeric_labels is not None
assert len(set(wrapper.numeric_labels)) == 229
assert math.isclose(Counter(wrapper.numeric_labels)[-1] / len(wrapper.numeric_labels) , 0.1, rel_tol = 0.05)

#### Test online model training

# IF 
# WHEN
wrapper.tm_setup()
wrapper.tm_train()

# THEN
assert wrapper.topics is not None

assert wrapper.topic_info is not None

assert len(wrapper.topics) == wrapper.subset.shape[0]
assert wrapper.topics_over_time is not None

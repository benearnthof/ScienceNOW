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

#### TEST SETUP
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
    "mask_probability": 0,
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 52 # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
}

model_type = "dynamic"

# WHEN
wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)

# THEN
assert wrapper.subset.shape == (49380, 17)
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
assert wrapper.probs is None
assert wrapper.topic_info is None
assert wrapper.topics_over_time is None

#### Test topic model setup
# IF
# WHEN
wrapper.tm_setup()

# THEN
assert isinstance(wrapper.umap_model, Dimensionality) 
assert wrapper.umap_model.reduced_embeddings.shape == (wrapper.subset.shape[0], wrapper.processor.PARAMS.UMAP_COMPONENTS.value)
assert isinstance(wrapper.hdbscan_model, HDBSCAN)
assert wrapper.ctfidf_model is not None
assert wrapper.topic_model is not None

#### Test topic model training

# IF
# WHEN
wrapper.tm_train()

# THEN
assert wrapper.topics is not None
assert wrapper.probs is not None
assert wrapper.topics_over_time is not None
assert wrapper.topic_info is not None

assert len(wrapper.topics) == wrapper.subset.shape[0]
assert len(wrapper.probs) == wrapper.subset.shape[0]
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
    "nr_bins": 52 # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
}

model_type = "semisupervised"

# WHEN
wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)

# THEN
assert wrapper.plaintext_labels is not None
assert wrapper.numeric_labels is not None
assert len(set(wrapper.numeric_labels)) == 78
assert math.isclose(Counter(wrapper.numeric_labels)[-1] / len(wrapper.numeric_labels) , 0.1, rel_tol = 0.015)

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

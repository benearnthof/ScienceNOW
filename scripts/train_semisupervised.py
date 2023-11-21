"""
Minimal script to train a semisupervised dynamic model
"""
from sciencenow.models.train import ModelWrapper

from pandas import Timestamp
from collections import Counter
import numpy as np
import math


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

model_type = "semisupervised"

# Init Wrapper class
wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)
# Model Setup and Training
wrapper.tm_setup()
wrapper.tm_train()

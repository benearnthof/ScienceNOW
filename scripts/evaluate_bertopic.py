"""
Demonstrating how to setup and evaluate BERTopic models automatically.
"""
# import time
# from tqdm import tqdm
# from sciencenow.models.train import ModelWrapper
# import Levenshtein 
# import numpy as np


# # quick evaluation run with only one model
# setup_params = {
#     "samples": 30, # hdbscan samples
#     "cluster_size": 1, # hdbscan minimum cluster size
#     "startdate": "01 01 2021", # if no date range should be selected set startdate to `None`
#     "enddate":"31 01 2021",
#     "target": "cs", # if no taxonomy filtering should be done set target to `None`
#     "threshold": 10,
#     "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
#                                 # contain labels not present in the first data set this to a data subset.
#     "mask_probability": 0,
#     "recompute": True,
#     "nr_topics": None,
#     "nr_bins": 4, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
#     "nr_chunks": 12, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
#     "evolution_tuning": False, # For dynamic model
#     "global_tuning": False, # For dynamic model
# }

# model_type = "semisupervised"

# # WHEN
# # start = time.time()
# wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)
# results = wrapper.train_and_evaluate()

# results[0]["Scores"]
# topics_over_time = wrapper.topics_over_time
# wrapper.get_dynamic_topics(wrapper.topics_over_time)

import time
from tqdm import tqdm
from sciencenow.models.train import ModelWrapper
import Levenshtein 
import numpy as np
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-clust", "--clust", type=int, help="HDBSCAN minimum cluster size")
args=parser.parse_args()
print(args.clust)

# results = []
errors = []

setup_params = {
    "samples": 30, # hdbscan samples
    "cluster_size": args.clust, # hdbscan minimum cluster size
    "startdate": "01 01 2021", # if no date range should be selected set startdate to `None`
    "enddate":"31 01 2021",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "threshold": 10,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": 0,
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 4, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": 12, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": False, # For dynamic model
    "global_tuning": False, # For dynamic model
}
wrapper = None
wrapper = ModelWrapper(setup_params=setup_params, model_type="semisupervised")
try:
    res = wrapper.train_and_evaluate(save=f"/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/tm_evaluation/{args.clust}")
    # results.append(res)
    del wrapper
    gc.collect()
except:
    print(f"An exception occurred at cluster size: {args.clust}")
    errors.append(args.clust)
    del wrapper
    gc.collect()

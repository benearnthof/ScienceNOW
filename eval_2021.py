"""
Demonstrating how to setup and evaluate BERTopic models automatically.
"""

import time
from tqdm import tqdm
from sciencenow.models.train import ModelWrapper 
import numpy as np
import gc
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-clust", "--clust", type=int, help="HDBSCAN minimum cluster size")
parser.add_argument("-sdate", "--sdate", type=str, help="startdate")
parser.add_argument("-edate", "--edate", type=str, help="enddate")
parser.add_argument("-dir", "--dir", type=str, help="output directory")
args=parser.parse_args()
print(args.clust)


setup_params = {
    "samples": 1, # hdbscan samples
    "cluster_size": args.clust, # hdbscan minimum cluster size
    "startdate": str(args.sdate), #"01 01 2021", # if no date range should be selected set startdate to `None`
    "enddate": str(args.edate), #"31 01 2021",
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
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 4, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": 4, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": False, # For dynamic model
    "global_tuning": False, # For dynamic model
    "limit": None,
    "subset_cache": "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/cache/",
}

online_params = {# For online DBSTREAM https://riverml.xyz/latest/api/cluster/DBSTREAM/
    "clustering_threshold": 1.0, # radius around cluster center that represents a cluster
    "fading_factor": 0.01, # parameter that controls importance of historical data to current cluster > 0
    "cleanup_interval": 2, # time interval between twwo consecutive time periods when the cleanup process is conducted
    "intersection_factor": 0.3, # area of the overlap of the micro clusters relative to the area cover by micro clusters
    "minimum_weight": 1.0 # minimum weight for a cluster to be considered not "noisy" 
}
    
print(setup_params["startdate"])
wrapper = None
wrapper = ModelWrapper(setup_params=setup_params, model_type="semisupervised")

#wrapper.tm_setup()
#wrapper.tm_train()
#wrapper.topic_info
try:
    res = wrapper.train_and_evaluate(save=f"/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/tm_evaluation/{args.dir}/{args.clust}")
#    res = wrapper.train_and_evaluate(save=f"/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/tm_evaluation/Jan2021/test")
#     # results.append(res)
#     # del wrapper
#     # gc.collect()
except:
    print(f"An exception occurred at cluster size: {args.clust}")
    del wrapper
    gc.collect()

# try subprocess to combat memory leak
# todo: write slurms for easy submission of evaluation
# investigate optimal cluster size 
#       are there similar paper clusters for different cluster sizes?
#       do papers tend to "stick together"?
# investigate influence of threshold
# investigate influence of mask probability

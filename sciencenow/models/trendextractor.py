"""Trend Extractor Class to postprocess a Topic Model"""
import pickle
import pandas as pd
import numpy as np
# from bertopic import BERTopic
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import re
import string
from dateutil import parser

#### PREPROCESSING

ROOT = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/")
ROOT.exists()
ARXIV_PATH = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/arxivdata/arxiv-metadata-oai-snapshot.json")
ARXIV_PATH.exists()
ARTIFACTS = ROOT / Path("artifacts/trends")
ARTIFACTS.exists()

# available from preprocessing: docs_2020 & timestamps_2020
docs_2020[0]
timestamps_2020[0]

data = docs_2020
timestamps = timestamps_2020
# we fit a topic model with the clustering hyperparameters that minimize number of outliers

from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
#_, timestamps = data_loader.load_docs()
#

embeddings = sentence_model.encode(data, show_progress_bar=True)

# precompute reduced embeddings
class Dimensionality:
    """ Use this for pre-calculated reduced embeddings """
    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings
    def fit(self, X):
        return self
    def transform(self, X):
        return self.reduced_embeddings

from umap import UMAP
umap_model = UMAP(n_neighbors = 15, n_components=5, metric='cosine', low_memory=False, random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

# hyperparameters that minimized number of outlieres: 
# min_cluster_size=225, min_samples=1

from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN

params = {
    "verbose": True,
    "umap_model": Dimensionality(reduced_embeddings),
    "hdbscan_model": HDBSCAN(min_cluster_size=225, min_samples=1, metric='euclidean', prediction_data=True),
    "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True),
    #"y": numeric_labels_2020
}


from bertopic import BERTopic
tm = BERTopic(**params)

topics, probs = tm.fit_transform(data, reduced_embeddings)
topics_over_time = tm.topics_over_time(data, timestamps, nr_bins=52)
topics_over_time.keys()

hierarchy = tm.hierarchical_topics(data)

hier_viz = tm.visualize_hierarchy(hierarchical_topics=hierarchy)
hier_viz.write_image(str(ARTIFACTS) + "/trendextractor_hier.png")

from typing import List
from pandas import DataFrame, Timestamp
import numpy as np


class TrendExtractor:
    def __init__(self,
                docs:List[str],
                timestamps:List[Timestamp],
                topics_over_time:DataFrame):
        self.docs = docs
        self.timestamps=timestamps
        self.topics_over_time=topics_over_time
    def extract_trends(self,window=3, threshold=1):
        """
        Every Topic has 52 Timestamps corresponding to the intervals of interest.
        We analyze the frequency of each topic in each timestamp
        """
        topicset = set(self.topics_over_time["Topic"])
        trends = []
        slopes = []
        for topic in topicset:
            topic_info = self.topics_over_time.loc[self.topics_over_time["Topic"] == topic]
            step_mean = np.mean(topic_info["Frequency"])
            topic_info["LTrend"] = self.map_counts(topic_info["Frequency"], window=3)
            # calculate differences from global mean + linear slope
            global_slope = self.get_global_slope(topic_info["Frequency"])
            slopes.append(global_slope)
            expected_counts = step_mean * np.ones(len(topic_info["Frequency"]))
            item_diffs = [i * global_slope for i, count in enumerate(expected_counts)]
            topic_info["GCounts"] = expected_counts + item_diffs
            topic_info["GDiffs"] = topic_info["Frequency"] - topic_info["GCounts"]
            topic_info["GTrend"] = self.get_global_trends(data=topic_info["GDiffs"], threshold=threshold)
            trends.append(topic_info)
        return(trends, slopes)
    @staticmethod
    def get_global_slope(data, order=1):
        """Obtain simple linear fit for global trend over all timesteps"""
        index = list(range(len(data)))
        coeffs = np.polyfit(index, list(data), order)
        slope = coeffs[-2] # global linear trend, polyfit returns coefficients order of descending degree
        return(float(slope))
    @staticmethod
    def map_counts(counts, window):
        trends = counts.rolling(window=window).mean().diff().fillna(0)
        return(trends)
    @staticmethod
    def get_global_trends(data, threshold):
        sd = np.std(data)
        up = data > (sd * threshold)
        down = data < (-sd * threshold)
        trends = ["Up" if x else "Down" if y else "Flat" for x, y in zip(up, down)]
        return (trends)


extractor = TrendExtractor(docs=data, timestamps=timestamps, topics_over_time=topics_over_time)
trends, slopes = extractor.extract_trends()


# TODO: compare to online model
# TODO: can we find clusters of documents in the evaluation topic results?
# TODO: rerun eval with semi supervised averaged embeddings approach

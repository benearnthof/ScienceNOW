"""
Setting up a basic dynamic model to demonstrate weak signal extraction according to the ideas presented in 
https://sci-hub.se/10.1016/j.eswa.2012.04.059
"""
from sciencenow.models.train import ModelWrapper

from pandas import Timestamp, DataFrame, concat
from scipy.stats import gmean
from collections import Counter
from tqdm import tqdm
from typing import Dict
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
    "nr_bins": 12 # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
}

model_type = "semisupervised"

# Init Wrapper class
wrapper = ModelWrapper(setup_params=setup_params, model_type=model_type)
# Model Setup and Training 
# Takes about 1 minute since the topic representations need to be recomputed for every bin.
wrapper.tm_setup()
wrapper.tm_train()

topics_over_time = wrapper.topics_over_time
topic_info = wrapper.topic_info
print(topic_info)
# 125 different topics, 17k outliers

def calculate_degree_of_diffusion(topics_over_time, timeweight=0.05):
    """
    Calculate Degree of Diffusion for a topic over all time intervals.
    https://sci-hub.se/10.1016/j.eswa.2012.04.059
    Degree of visibility (DoV) of topic i in period j can be defined as:
    DoV_{ij} = (TF_{ij}/NN_{j}) * (1 - tw * (n-j))
    Relative Term Frequency * (1 - 0.05 * (52 - Periodindex))
    Params: 
        topics_over_time: data frame that contains Frequency of all topics for all time intervals of interest
        timeweight: factor that influences how much trends are biased to the most recent date (enddate) 
            chosen prior tho the model fitting. Default: 0.05 in accordance with 
            https://sci-hub.se/10.1016/j.eswa.2012.04.059
    """
    # TODO: Very important Question: 
    # The Time weight factor biases this measure towards the most recent publications
    # in our case the "Enddate" chosen for the analysis. 
    # How do we pick this value to make sure we still find trends in the recent past?
    topicset = set(topics_over_time.Topic)
    # we need the total number of publications in each timestamp
    timestamps = set(topics_over_time.Timestamp)
    frequencies_by_timestamp = topics_over_time.groupby("Timestamp").sum()
    freqs = np.array(frequencies_by_timestamp.Frequency.tolist())
    time_factors = np.array([1 - timeweight * (len(freqs) - j) for j in range(0, len(freqs), 1)])
    results = {}
    for topic in tqdm(topicset):
        topic_data = topics_over_time.loc[topics_over_time.Topic == topic]
        topic_timestamps = set(topic_data.Timestamp)
        missing_timestamps = timestamps.difference(topic_timestamps)
        # need to add empty lines to data frames to match lengths for broadcasting
        if missing_timestamps:
            for i, ts in enumerate(missing_timestamps):
                line = DataFrame({"Topic": topic, "Words": "default", "Frequency": 0, "Timestamp": ts}, index = [0.5])
                topic_data = concat([topic_data, line])
        topic_data = topic_data.sort_values(by=["Timestamp"])
        assert len(topic_data) == len(freqs)
        topic_freqs = np.array(topic_data.Frequency.tolist())
        relative_topic_freqs = topic_freqs / freqs
        deg_of_diffusion= relative_topic_freqs * time_factors
        results[topic] = deg_of_diffusion
    return results

results = calculate_degree_of_diffusion(topics_over_time)

def calculate_topic_issue_map(topics_over_time):
    """
    Calculates geometric mean of degrees of diffusion for every topic and the respective 
    average document frequencies
    """
    topic_set = set(topics_over_time.Topic)
    degs_of_diffusion = calculate_degree_of_diffusion(topics_over_time)
    geometric_means = {topic: gmean(degs_of_diffusion[topic]) for topic in topic_set}
    growth_rates = {
        topic: 
        np.mean( # pairwise differences
            [j - i for i, j in zip(degs_of_diffusion[topic][: -1], degs_of_diffusion[topic][1 :])]
            ) 
        for topic in topic_set
        }
    average_frequencies = {
        topic: 
        topics_over_time
        .loc[topics_over_time.Topic == topic]
        .groupby("Topic")
        .mean()
        .Frequency
        .tolist()[0] 
        for topic in topic_set
        }
    return geometric_means, average_frequencies, growth_rates

gmeans, avg_freqs, grates = calculate_topic_issue_map(topics_over_time)

temp = DataFrame({
        "Topic": list(set(topics_over_time.Topic)),
        "DoD_gmean": list(gmeans.values()),
        "Avg_Freq": list(avg_freqs.values()),
        "Growth_Rate": list(grates.values())
        })

temp = temp.sort_values(by=["Topic"])

# topic_info is already sorted by topic, we can just append the two additional columns
tinfo = topic_info.assign(DoD_gmean=temp.DoD_gmean.tolist())
tinfo = tinfo.assign(Average_Frequency=temp.Avg_Freq.tolist())
tinfo = tinfo.assign(Growth_Rate=temp.Growth_Rate.tolist())

# degree of visibility = total occurence of keyword per time period
# degree of diffusion = document frequency

ax1 = tinfo.plot.scatter(
    x="Average_Frequency",
    y="Growth_Rate",
)

fig = ax1.get_figure()

fig.savefig("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/weak_signals.png")

test = [0.218, 0.225, 0.269, 0.242, 0.276, 0.318, 0.330, 0.302, 0.359, 0.387, 0.395]
# target "increasing rate" reported in paper
np.mean(test)
increasing_rate = 0.061

# rate of change from i to j
res = [j - i for i, j in zip(test[: -1], test[1 :])]
np.mean(res)
gmean(res)

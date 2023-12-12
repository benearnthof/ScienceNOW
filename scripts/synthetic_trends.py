"""
Analysis of the impact of new "artificial" data on the model performance.
We're specifically looking for threhold values for the amount of papers in 
a topic needed for the model to pick it up as an emerging trend.
"""

from sciencenow.postprocessing.trends import TrendExtractor
from sciencenow.models.train import ModelWrapper

setup_params = {
    "samples": 1, # hdbscan samples
    "cluster_size": 25, # hdbscan minimum cluster size
    "startdate": "01 01 2021", # if no date range should be selected set startdate to `None`
    "enddate": "31 01 2021",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "secondary_target": "q-bio", # for synthetic trend extraction
    "secondary_startdate": "01 01 2020",
    "secondary_enddate": "31 12 2020",
    "secondary_proportion": 0.2,
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
}
# print(setup_params["startdate"])
#wrapper = None
wrapper = ModelWrapper(setup_params=setup_params, model_type="semisupervised")
# wrapper.subset.v1_datetime

# idea: we use subset of q-bio papers form 2020 (2858 papers in total)
# to test if the dynamic or online models are able to pick out these artificial trends
# why 2020? 
# because if we use papers published earlier than 2021 for example
# we make sure that our "artificial trend papers" could not possibly
# have cited the 2021 cs papers we're interested in. 
# Make sure there is only little chance for the papers in both respective sets to be related.

wrapper.tm_setup()
out = wrapper.tm_train()

#model1 = wrapper.topic_model
#model2 = wrapper2.topic_model

# from bertopic import BERTopic
# merged_model = BERTopic.merge_models([model1, model2])
# merging only works if we train on different subsets, defeating the purpose of merging
# the subsets to begin with

# hierarchy = wrapper.topic_model.hierarchical_topics(wrapper.subset.abstract.tolist())

# hier_viz = wrapper.topic_model.visualize_hierarchy(hierarchical_topics=hierarchy)
# hier_viz.write_image("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4" + "/no-bio_hierarchy.png")

extractor = TrendExtractor(model_wrapper=wrapper)
deviations = extractor.calculate_deviations()

candidates = extractor.get_candidate_papers(
    subset=wrapper.subset,
    topics=wrapper.topics,
    deviations=deviations,
    threshold=1.5)

# # papers = [x[0] for x in candidates[17]]
labels = {}
for key in candidates:
    labs = [x[3] for x in candidates[key]]
    labels[key] = labs

from collections import Counter
counters = []
for key in labels:
    counters.append(Counter(labels[key]).most_common())




# means, avg_freqs, grates = extractor.extract_weak_signals()

# from pandas import DataFrame
# temp = DataFrame({
#         "Topic": list(set(wrapper.topics_over_time.Topic)),
#         "DoD_mean": list(means.values()),
#         "Avg_Freq": list(avg_freqs.values()),
#         "Growth_Rate": list(grates.values())
#         })

# temp = temp.sort_values(by=["Topic"])


# tinfo = wrapper.topic_info

# # topic_info is already sorted by topic, we can just append the two additional columns
# tinfo = tinfo.assign(DoD_mean=temp.DoD_mean.tolist())
# tinfo = tinfo.assign(Average_Frequency=np.log(temp.Avg_Freq.tolist()))
# tinfo = tinfo.assign(Growth_Rate=temp.Growth_Rate.tolist())

# tinfo = tinfo[tinfo.Topic != -1]


# # degree of visibility = total occurence of keyword per time period
# # degree of diffusion = document frequency

# ax1 = tinfo.plot.scatter(
#     x="Average_Frequency",
#     y="Growth_Rate",
# )

# fig = ax1.get_figure()

# fig.savefig("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/weak_signals_2021.png")

# we are able to find conferences in the data
# http://ifiptc9.org/wg94/ifip-9-4-conferences/ifip-9-4-virtual-conference-2021/

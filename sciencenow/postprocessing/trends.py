import time
from tqdm import tqdm
import numpy as np
from typing import List
from pandas import Timestamp, DataFrame, concat
import datetime
from scipy.stats import gmean


class TrendExtractor:
    """
    Post processing class we initialize with a model_wrapper that has already been trained.
    We use the document subset, the respective timestamps & the topics over time obtained during
    training to extract trends and highlight emergent fields.
    """
    def __init__(self, model_wrapper):
        self.docs = model_wrapper.subset.abstract.tolist()
        self.timestamps = model_wrapper.subset.v1_datetime.tolist()
        self.topics_over_time = model_wrapper.topics_over_time

    def extract_linear_trends(self, window=1, threshold=1):
        """
        Every Topic has a number of Timestamps corresponding to the intervals of interest.
        We analyze the frequency of each topic in each timestamp
        """
        topicset = set(self.topics_over_time["Topic"])
        trends = []
        slopes = []
        for topic in topicset:
            topic_info = self.topics_over_time.loc[self.topics_over_time["Topic"] == topic]
            step_mean = np.mean(topic_info["Frequency"])
            topic_info["LTrend"] = self.map_counts(topic_info["Frequency"], window=1)
            # calculate differences from global mean + linear slope
            global_slope = self.get_global_slope(topic_info["Frequency"])
            slopes.append(global_slope)
            expected_counts = step_mean * np.ones(len(topic_info["Frequency"]))
            item_diffs = [i * global_slope for i, count in enumerate(expected_counts)]
            topic_info["GCounts"] = expected_counts + item_diffs
            topic_info["GDiffs"] = topic_info["Frequency"] - topic_info["GCounts"]
            topic_info["GTrend"] = get_global_trends(data=topic_info["GDiffs"], threshold=threshold)
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

    def calculate_deviations(self):
        """
        Simplest Approach: If the topic count drastically differs from the mean of 
        counts over all time stamps we mark it as trending.
        """
        topicset = set(self.topics_over_time["Topic"])
        trends = []
        for topic in topicset:
            topic_info = self.topics_over_time.loc[self.topics_over_time["Topic"] == topic]
            topic_mean = np.mean(topic_info["Frequency"])
            topic_sd = np.std(topic_info["Frequency"])
            topic_info["Difference"] = topic_info["Frequency"] - topic_mean
            topic_info["Sigma"] = topic_info["Difference"] / topic_sd
            trends.append(topic_info)
        return trends

    @staticmethod
    def get_candidate_papers(subset, topics, deviations, threshold=3, delta=datetime.timedelta(days=7)):
        """
        Method to filter papers by timeframes of potentially trending topics
        """
        candidate_topic_indices = []
        for i, deviation in enumerate(deviations):
            if any(deviation["Sigma"] >= threshold):
                candidate_topic_indices.append(i)

        # papers = subset.abstract.tolist()
        indices = subset.index.tolist()
        ids = subset.id.tolist()
        titles = subset.title.tolist()
        l1_labels = subset.l1_labels.tolist()
        plaintext_labels = subset.plaintext_labels.tolist()
        timestamps = subset.v1_datetime.tolist()
        # Return ID instead of abstract
        # ID to select embeddings for visualization
        # Title for Title highlighting in 3D plot

        candidates = {}
        for candidate_topic in candidate_topic_indices:
            # t_papers = [paper for paper, topic in zip(papers, topics) if topic == candidate_topic]
            t_l1 = [label for label, topic in zip(l1_labels, topics) if topic == candidate_topic]
            t_plaintext = [plain for plain, topic in zip(plaintext_labels, topics) if topic == candidate_topic]
            t_timestamps = [ts for ts, topic in zip(timestamps, topics) if topic == candidate_topic]
            t_indices = [ts for ts, topic in zip(indices, topics) if topic == candidate_topic]
            t_ids = [ts for ts, topic in zip(ids, topics) if topic == candidate_topic]
            t_titles = [ts for ts, topic in zip(titles, topics) if topic == candidate_topic]
            begins = deviations[candidate_topic][deviations[candidate_topic]["Sigma"] >= threshold]
            ends = begins["Timestamp"] + delta
            t_candidates = []
            for begin, end in zip(begins["Timestamp"].tolist(), ends.tolist()):
                # print(begin, end)
                cands = [
                    {
                        "timestamp":ts,
                        "l1_label":l1,
                        "plaintext":plain,
                        "index":index,
                        "id":ids,
                        "title":title,
                    }
                    for ts, l1, plain, index, ids, title in 
                    zip(t_timestamps, t_l1, t_plaintext, t_indices, t_ids, t_titles) if 
                    ts >= begin and ts <= end
                    ]
                t_candidates.extend(cands)
            candidates[candidate_topic] = t_candidates

        return candidates

    def calculate_degree_of_diffusion(self, timeweight=0.019):
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
        topicset = set(self.topics_over_time.Topic)
        # we need the total number of publications in each timestamp
        timestamps = set(self.topics_over_time.Timestamp)
        frequencies_by_timestamp = self.topics_over_time.groupby("Timestamp").sum()
        freqs = np.array(frequencies_by_timestamp.Frequency.tolist())
        time_factors = np.array([1 - timeweight * (len(freqs) - j) for j in range(0, len(freqs), 1)])
        results = {}
        for topic in tqdm(topicset):
            topic_data = self.topics_over_time.loc[self.topics_over_time.Topic == topic]
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

    def extract_weak_signals(self):
        """
        Weak singnals approach to nowcast current trends.
        Calculates geometric mean of degrees of diffusion for every topic and the respective 
        average document frequencies
        """
        topic_set = set(self.topics_over_time.Topic)
        degs_of_diffusion = self.calculate_degree_of_diffusion()
        # geometric_means = {topic: gmean(degs_of_diffusion[topic]) for topic in topic_set}
        means = {topic: np.mean(degs_of_diffusion[topic]) for topic in topic_set}
        growth_rates = {
            topic: 
            np.mean( # pairwise differences
                [j - i for i, j in zip(degs_of_diffusion[topic][: -1], degs_of_diffusion[topic][1 :])]
                ) 
            for topic in topic_set
            }
        average_frequencies = {
            topic: 
            self.topics_over_time
            .loc[self.topics_over_time.Topic == topic]
            .groupby("Topic")
            .mean()
            .Frequency
            .tolist()[0] 
            for topic in topic_set
            }
        return means, average_frequencies, growth_rates



#extractor = TrendExtractor(docs=data, timestamps=timestamps, topics_over_time=topics_over_time)
#trends, slopes = extractor.extract_trends()


# TODO: compare to online model
# TODO: can we find clusters of documents in the evaluation topic results?

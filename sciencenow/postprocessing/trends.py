import time
import warnings
from tqdm import tqdm
import numpy as np
from pandas import DataFrame, concat
import datetime
from typing import List, Set, Dict, Tuple
from math import log2



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

    def extract_linear_trends(self, threshold=1):
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
    def get_global_trends(data, threshold) -> List[str]:
        sd = np.std(data)
        up = data > (sd * threshold)
        down = data < (-sd * threshold)
        trends = ["Up" if x else "Down" if y else "Flat" for x, y in zip(up, down)]
        return (trends)

    def calculate_deviations(self) -> List[DataFrame]:
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
    def get_candidate_papers(subset, topics, deviations, threshold=3, delta=datetime.timedelta(days=7)) -> Dict[Dict]:
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

    def extract_weak_signals(self) -> Tuple[Dict, Dict, Dict]:
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


class TrendValidatorIR:
    """
    Class that calculates precision & Discounted Cumulative gain for assessment of synthetic Trends.
    Based on Ideas from Information Retrieval. 
    """
    def __init__(
        self,
        results: Dict[int, Set[str]], # candidate_ids
        background_docs: Set[str], # background_ids
        synth_background_docs: Set[str], # synthetic background ids
        target_docs: Set[str], # target_ids
    ):   
        """
        Initialize the Trend Validator

        Args:
            results: A Dictionary that contains sets of ids of papers that were found to be trending in any time frame. Their Keys correspond to the trend ids of BERTopic
            background_docs: The set of ids of documents that were non synthetic
            synth_background_docs: The set of ids of synthetic documents that fell into a time frame not designated as synthetic trend
            target_docs: The set of ids of synthetic documents that fell into the designated trend time frame
        """
        self.results = [x for x in results.values()]
        self.background_docs = background_docs
        self.synth_background_docs = synth_background_docs
        self.target_docs = target_docs

    def precision_at_k(self, k: int, threshold: float = 0.5) -> float:
        """
        Calculates the precision at k for the results.

        Parameters:
        - k (int): The number of results to consider.
        - threshold (float): The threshold value for trend validation.

        Returns:
        - float: The precision at k.
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0

        for rank in range(k):
            trend = self.results[rank]
            if self._validate_trend(trend, threshold) == "tp":
                true_positives += 1
            elif self._validate_trend(trend, threshold) == "fp":
                false_positives += 1
            else:
                true_negatives += 1

        total = true_positives + false_positives
        if total == 0:
            return 0
        else:
            return true_positives / (true_positives + false_positives)

    def dcg_at_k(self, k: int, normalize=False) -> float:
        """
        Discounted cumulative gain (DCG) at k.

        Parameters:
        k (int): The value of k.

        Returns:
        float: The DCG value at k.
        """

        dcg = 0
        # loop through each item and calculate DCG
        for rank in range(k):
            trend = self.results[rank]
            rel_k = self._get_trend_fraction(trend)
            # calculate DCG
            dcg += rel_k / log2(1 + rank + 1)

        # normalize DCG
        if normalize:
            dcg /= self.__ideal_dcg_at_k(k)

        return dcg

    @staticmethod
    def __ideal_dcg_at_k(k: int) -> float:
        """
        Calculates the ideal DCG at k.

        Parameters:
        k (int): The value of k.

        Returns:
        float: The ideal DCG value at k.
        """
        return sum(1 / log2(1 + i + 1) for i in range(k))

    def _validate_trend(self, trend: Set[str], threshold: float) -> bool:
        """
        Validates a trend by comparing the trend with the target documents.

        Args:
            trend (Set[str]): The trend to be validated.
            threshold (float): The threshold value for the precision.

        Returns:
            bool: True if the trend is validated, False otherwise.
        """
        # Get the intersection of the trend and the target documents
        target_intersection = trend.intersection(self.target_docs)
        background_intersection = trend.intersection(self.background_docs)
        synth_intersection = trend.intersection(self.synth_background_docs)
        # Calculate the precision
        tp_fraction = len(target_intersection) / len(trend)
        bg_fraction = len(background_intersection) / len(trend)
        fp_fraction = len(synth_intersection) / len(trend)
        # if target intersection is not empty then we did detect a target trend
        # if synth intersection is empty and target intersection is empty we have a background trend that should not have an impact
        # if synth intersection is non-empty we have a false positive
        if tp_fraction >= threshold:
            return "tp"
        elif bg_fraction >= threshold:
            return "tn"
        elif fp_fraction >= threshold:
            return "fp"
        # # Return True if the precision is greater than the threshold
        # return fraction >= threshold

    def _get_trend_fraction(self, trend: Set[str]) -> bool:
        """
        Validates a trend by comparing the trend with the target documents.
        Returns the trend fraction.

        Parameters:
        - trend (Set[str]): The trend to be validated.

        Returns:
        - bool: The trend fraction.

        """
        # Get the intersection of the trend and the target documents
        intersection = trend.intersection(self.target_docs)

        # Calculate the precision
        fraction = len(intersection) / len(trend)

        # Return True if the precision is greater than the threshold
        return fraction


class TrendPostprocessor:
    """
    Class that uses Extractor and Validator to extract trending paper IDs from corpus.
    """
    def __init__(self, wrapper) -> None:
        """
        Initialize the Trend Postprocessor

        Args:
            wrapper: A ModelWrapper that has been trained and from which we can potentially extract trends.
            threshold
        """
        if wrapper.topics_over_time is None:
            raise NotImplementedError("Cannot extract trends from untrained model.")
        self.extractor = TrendExtractor(model_wrapper=wrapper)
        self.subset = wrapper.subset
        self.topics = wrapper.topics
        self.topic_info = wrapper.topic_info
        self.embeddings = wrapper.subset_reduced_embeddings
        self.deviations = self.extractor.calculate_deviations()
        self.papers_per_bin = wrapper.papers_per_bin
        self.target = wrapper.setup_params["secondary_target"]
        if self.target is not None:
            self.ds_synthetic = self.subset[self.subset["l1_labels"].str.startswith(f"{self.target}")]
        else:
            self.ds_synthetic = None
        # pandas: ~ to invert boolean series
        self.ds_background = self.subset[~self.subset["l1_labels"].str.startswith(f"{self.target}")]
        self.validator = None

    def get_candidate_papers(self, threshold):
        """
        Wrapper method that allows us to pass an arbitrary threshold to 
        self.extractor.get_candidate_papers

        Args:
            threshold: Number that specifies how strictly a trend is defined.
        """
        candidates = self.extractor.get_candidate_papers(
            subset=self.subset,
            topics=self.topics,
            deviations=self.deviations,
            threshold=threshold
        )
        return candidates

    def extract_max_papers(self):
        """
        Helper that first extracts the synthetic papers from the subset and then returns only those
        rows that were designated as "Trending". => Those rows that fell into the largest bin containing the 
        artificial influx of papers.
        """
        cumulative_sums = np.cumsum(self.papers_per_bin)
        max_index = np.argmax(self.papers_per_bin)
        if max_index == 0:
            start_index = 0
        else:
            start_index = cumulative_sums[max_index - 1]
        end_index = cumulative_sums[max_index]
        return self.ds_synthetic.iloc[start_index:end_index]
        

    @staticmethod
    def extract_ids(dictionary: Dict[List]) -> Dict[Set]:
        """
        Will convert a nested dictionary to a dictionary of sets.
        """
        id_dict = {}
        for key, list_of_dicts in dictionary.items():
            for item in list_of_dicts:
                if key not in id_dict:
                    id_dict[key] = []
                id_dict[key].append(item['id'])
        return {key: set(value) for key, value in id_dict.items()}
    
    @staticmethod
    def compute_union(dictionary_of_sets):
        """
        Compute finite union of countably many sets.
        """
        union_set = set()
        for s in dictionary_of_sets.values():
            union_set |= s
        return union_set

    def calculate_performance(self, threshold) -> Dict[List, List, float]:
        """
        Method that uses TrendValidatorIR to calculate precisions & DCG
        Will only run successfully if a synthetic trend has been added to data, else first line returns None
        """
        if self.ds_synthetic is None:
            warnings.warn("No synthetic trends found, skipping performance calculation...")
            return None
        target_set = self.extract_max_papers(self.ds_synthetic, self.papers_per_bin)
        target_ids = set(target_set.id.tolist())
        candidates = self.get_candidate_papers(threshold=threshold)
        candidate_ids = self.extract_ids(candidates)
        background_ids = set(self.ds_background.id.tolist()).difference(target_ids)
        synth_background_ids = set(self.ds_synthetic.id.tolist()).difference(target_ids)
        self.validator = TrendValidatorIR(
            results=candidate_ids,
            background_docs=background_ids,
            synth_background_docs=synth_background_ids,
            target_docs=target_ids
        )
        # calculate precisions & discounted cumulative gains
        precisions = [self.validator.precision_at_k(k) for k in range(len(candidate_ids))]
        dcg_at_ks = [self.validator.dcg_at_k(k) for k in range(len(candidate_ids))]
        # overall performance: How many target ids could be found in trending clusters
        trending_ids = self.compute_union(candidate_ids)
        trend_intersection = trending_ids.intersection(target_ids)
        overall_performance = len(trend_intersection) / len(target_ids)
        return {"precisions":precisions, "dcg": dcg_at_ks, "overall_performance": overall_performance}
    
    def get_trend_info(self, threshold):
        """
        Method that extracts relevant info for trending papers like Topic & Title
        """
        candidates = self.get_candidate_papers(threshold=threshold)
        candidate_ids = self.extract_ids(candidates)
        trending_ids = self.compute_union(candidate_ids)
        df = self.subset
        df["Topic"] = self.topics
        trend_df = df[df["id"].isin(trending_ids)]
        # make sure background topic is never trending (does not provide meaningful info)
        trend_df = trend_df[trend_df["Topic"] != -1]
        # Get reduced embeddings for each paper so we can visualize them
        trend_embeddings = self.embeddings[trend_df.index]
        # Get the topic representation so we can use them to label the plot
        topic_reps = [self.topic_info[self.topic_info["Topic"]==x].Representation.tolist()[0] for x in trend_df["Topic"].tolist()]
        # merge trend info with trending df
        trend_df["Representation"] = topic_reps
        return trend_df, trend_embeddings

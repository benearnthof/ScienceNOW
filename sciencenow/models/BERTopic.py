"""
Topic Modeling with BERTopic
"""

# the algo:
# embed documents
# umap for dimensionality reduction of embeddings
# hdbscan for clustering of reduced embeddings
# create topic representations from clusters (c-tf-idf) & maximize relevance

# bertopic uses sentence transformers
import pandas as pd
import numpy as np
from bertopic import BERTopic
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

ARXIV_PATH = Path("c:/arxiv/arxiv-metadata-oai-snapshot.json")


class ArxivProcessor:
    def __init__(self, path=ARXIV_PATH, sorted=True **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path
        self.sorted = sorted

    def _get_data(self):
        with open(self.path, "r") as file:
            for line in file:
                yield line

    def _process_data(self)
        date_format = "%a, %d %b %Y %H:%M:%S %Z"
        data_generator = self._get_data()
        ids, titles, abstracts, cats, refs, timestamps = [],[],[],[],[],[]
        for paper in tqdm(data_generator):
            paper_dict = json.loads(paper)
            ids.append(paper_dict["id"])
            titles.append(paper_dict["title"])
            abstracts.append(paper_dict["abstract"])
            categories.append(paper_dict["categories"])
            refs.append(paper_dict["journal-ref"])
            timestamps.append(paper_dict["versions"][0]["created"])  # 0 should be v1
        # process timestamps so that they can be sorted
        timestamps_datetime = [datetime.strptime(stamp, date_format) for stamp in timestamps]
        out = pd.DataFrame(
            {
                "id": ids,
                "title": titles,
                "abstract": abstracts,
                "categories": categories,
                "timestamp": timestamps_datetime,
            }
        )
        if self.sorted:
            return out.sort_values("timestamp", ascending=False)
        return out



#print(len(ids) / index)
# about 30% of the data

#for ref in refs[0:100]:
#    print(f"{ref}")

# about 60& of papers dont have a journal reference
# using timestamps of versions as a proxy for release year
for stamp in timestamps[-100:]:
    print(stamp)

None in timestamps  # false
# this seems to be a good solution

date_format = "%a, %d %b %Y %H:%M:%S %Z"
datetime_object = datetime.strptime(stamp, date_format)

timestamps_datetime = [datetime.strptime(stamp, date_format) for stamp in timestamps]

sorted_times = sorted(timestamps_datetime)  # TODO visualize data


# use subset for now
df = pd.DataFrame(
    {
        "id": ids,
        "title": titles,
        "abstract": abstracts,
        "categories": categories,
        "timestamp": timestamps_datetime,
    }
)

sorted = df.sort_values("timestamp", ascending=False)


df.head()
# inspect available categories
cat_list = df["categories"].unique()
print(len(cat_list))  # 6070 unique categories
cat_list[0:10]

# there should be some hierarchy to these clusters
# restricting ourselves to computer science related articles
cs_df = df[df["categories"].str.contains("cs.")]
print(len(cs_df))  # 13738 Articles are left over
print(len(cs_df["categories"].unique()))  # 2842 Categories are now left

sentence_list = cs_df["abstract"].tolist()
print(sentence_list[0])


# The Topic Modeling starts here
sample = sentence_list[0:2500]

topic_model = BERTopic(calculate_probabilities=True)
topics, probabilities = topic_model.fit_transform(sample)
# topics => indices, probs => array of floats

topic_info = topic_model.get_topic_info()

topic_model.get_topic(1)  # first "interesting" topic
topic_model.get_topic(10)

barchart = topic_model.visualize_barchart(top_n_topics=10)
barchart.show()

viz = topic_model.visualize_topics()
viz.show()

hierarchy = topic_model.visualize_hierarchy()
hierarchy.show()

heatmap = topic_model.visualize_heatmap(n_clusters=7)
heatmap.show()

probs = topic_model.visualize_distribution(probabilities[10])
probs.show()

####
# swapping out the dim reduction & clustering methods to improve speed
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

dim_model = PCA(n_components=5)
cluster_model = KMeans(n_clusters=50)

topic_model = BERTopic(
    umap_model=dim_model,
    embedding_model="allenai-specter",
    hdbscan_model=cluster_model,
    calculate_probabilities=True,
)

topics, probabilities = topic_model.fit_transform(sample)


barchart = topic_model.visualize_barchart(top_n_topics=10)
barchart.show()
viz = topic_model.visualize_topics()
viz.show()
hierarchy = topic_model.visualize_hierarchy()
hierarchy.show()
heatmap = topic_model.visualize_heatmap(n_clusters=3)
heatmap.show()


# Guided Topic Modeling:
# select list of seed topics
# map to original categories/clusters
seed_topic_list = [
    ["graph", "graphs", "problem", "polynomial"],
    ["logic", "semantics", "that", "of"],
    ["network", "networks", "community", "degree"],
]

topic_model = BERTopic(seed_topic_list=seed_topic_list)
# TODO: fix compatibility issue with python 3.10 (numpy 1.21.0 should work)
topics, probabilities = topic_model.fit_transform(sample)


# bertopic over time => topics_over_time
# for instances that are dynamic over time
# Evolution of topics over time
# calculate global topic model for largest topics that should be available at every time step
# then obtain time specific representations
# weighted average of global representation and repr. at t-1 allows "evolution" of
# repr over time
# => need twitter data for meaningful results


# If new data comes in we can use online modeling
# use incremental PCA for dimensionality reduction
# and minibatch KMeans for clustering
# (also need an incremental vectorizer)

# to continuously add new clusters we need the river package

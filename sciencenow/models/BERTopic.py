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

arxiv_path = Path("c:/arxiv/arxiv-metadata-oai-snapshot.json")


def get_data():
    with open(arxiv_path, "r") as f:
        for line in f:
            yield line


data = get_data()  # generator, emits strings
ids = []
titles = []
abstracts = []
categories = []
refs = []
timestamps = []
index = 0


for paper in data:
    index += 1
    paper_dict = json.loads(paper)
    try:
        #    try: #TODO: more general way to obtain year from ref
        #        year = int(
        #            paper_dict["journal-ref"][-4:]
        #        )  ### Example Format: "Phys.Rev.D76:013009,2007"
        #    except:
        #        year = int(
        #            paper_dict["journal-ref"][-5:-1]
        #        )  ### Example Format: "Phys.Rev.D76:013009,(2007)"
        ids.append(paper_dict["id"])
        titles.append(paper_dict["title"])
        abstracts.append(paper_dict["abstract"])
        categories.append(paper_dict["categories"])
        refs.append(paper_dict["journal-ref"])
        timestamps.append(paper_dict["versions"][0]["created"])  # 0 should be v1
    except:
        pass
    if index % 1000 == 0:
        print(index)

print(len(ids) / index)
# about 30% of the data

for ref in refs[0:100]:
    print(f"{ref}")

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
        "id": ids[0:100000],
        "title": titles[0:100000],
        "abstract": abstracts[0:100000],
        "categories": categories[0:100000],
    }
)

df.head()
# inspect available categories
cat_list = df["categories"].unique()
print(len(cat_list))  # 6070 unique categories
cat_list[0:10]

# there should be some hierarchy to these clusters
# restricting ourselves to computer science related articles
cs_df = df[df["categories"].str.contains("cs.")]
print(len(cd_df))  # 13738 Articles are left over
print(len(cs_df["categories"].unique()))  # 2842 Categories are now left

sentence_list = cs_df["abstract"].tolist()
print(sentence_list[0])


# The Topic Modeling starts here
sample = sentence_list[0:2500]

topic_model = BERTopic(calculate_probabilities=True)
topics, probs = topic_model.fit_transform(sample)
# topics => indices, probs => array of floats

topic_info = topic_model.get_topic_info()

topic_model.get_topic(0)  # first "interesting" topic
topic_model.get_topic(10)


# bertopic over time => topics_over_time
# for instances that are dynamic over time

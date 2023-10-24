"""
Trying to uncover Trends by observing relative cluster growth over time.
Construct synthetic dataset with fixed number of papers in each cluster.
Vary their respective dates of publication to simulate emerging trends
Use cluster scoring metrics to evaluate model performance
    We can include the Hierarchy of Topics for Arxiv (L0, L1) for Bertopic Model.
Models we Want to Benchmark:
    Bertopic Dynamic
    Bertopic Online
    ANTM
    Bertopic Hierarchical (if possible)
"""

#### Constructing Synthetic Datasets

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

dframe = pd.read_json(ARXIV_PATH, lines=True)
df = dframe

# creating three subsets for publications: 2020, 2021, 2022
paper = dframe[0:1]
versions = paper["versions"]
created = versions[0][0]["created"]
# 'Mon, 2 Apr 2007 19:18:42 GMT'
date = parser.parse(created)

# dframe["v1_date"] = [parser.parse(version[0]["created"]) for version in dframe["versions"] if version]
for version in df["versions"][0:100]:
    print(version[0]["created"])

v1_dates = [version[0]["created"] for version in df["versions"]]
v1_datetime = [parser.parse(dt) for dt in tqdm(v1_dates)]

df["v1_datetime"] = v1_datetime
df.sort_values("v1_datetime")
# now finally subset for the three years of interest
with open (str(ARTIFACTS) + "/df.pickle", "wb") as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

mask = (df["v1_datetime"] >= datetime.datetime(2020, 1, 1, 0, 0, 0) & df["v1_datetime"] <= parser.parse("2020-12-31"))
df["v1_datetime"] >= datetime(2020, 1, 1, 0, 0, 0, tzinfo="GMT")

start_2020 = pd.to_datetime(parser.parse("Wed, 1 Jan 2020 00:00:00 GMT"))
#end_2020 = parser.parse("Thu, 31, Dec 2020 23:59:59 GMT")
start_2021 = pd.to_datetime(parser.parse("Fri, 1 Jan 2021 00:00:00 GMT"))
#end_2021 = parser.parse("Fri, 31, Dec 2021 23:59:59 GMT")
start_2022 = pd.to_datetime(parser.parse("Sat, 1 Jan 2022 00:00:00 GMT"))
end_2022 = pd.to_datetime(parser.parse("Sat, 31, Dec 2022 23:59:59 GMT"))

mask_2020 = [(day > start_2020) & (day < start_2021) for day in tqdm(df["v1_datetime"])]
mask_2021 = [(day > start_2021) & (day < start_2022) for day in tqdm(df["v1_datetime"])]
mask_2022 = [(day > start_2022) & (day < end_2022) for day in tqdm(df["v1_datetime"])]


subset_2020 = df.loc[mask_2020]
subset_2021 = df.loc[mask_2021]
subset_2022 = df.loc[mask_2022]

subset_2020 = subset_2020.sort_values("v1_datetime")
subset_2021 = subset_2021.sort_values("v1_datetime")
subset_2022 = subset_2022.sort_values("v1_datetime")

# pickle all datasets so we don't have to preprocess everything every time we restart the session
with open (str(ARTIFACTS) + "/subset_2020.pickle", "wb") as handle:
    pickle.dump(subset_2020, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open (str(ARTIFACTS) + "/subset_2021.pickle", "wb") as handle:
    pickle.dump(subset_2021, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open (str(ARTIFACTS) + "/subset_2022.pickle", "wb") as handle:
    pickle.dump(subset_2022, handle, protocol=pickle.HIGHEST_PROTOCOL)

# about 300 megabytes each, seems workable

"""
Columns of Interest: 
    categories,
    abstract,
    v1_datetime,
"""
# we sorted by datetime, we can now split the papers into 52 chunks corresponding to each week 
# and then compare categories
# lets make a table of the categories that are present first
from collections import Counter
cat_counter = Counter(subset_2020["categories"])
# over 17000 distinct label combinations
# every paper can fall into multiple categories
# lets select just the computerscience papers

taxonomy = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/taxonomy.txt")

with open(taxonomy, "r") as file:
    arxiv_taxonomy = [line.rstrip() for line in file]

arxiv_taxonomy_list = [line.split(":") for line in arxiv_taxonomy]
arxiv_taxonomy = {line[0]: line[1].lstrip() for line in arxiv_taxonomy_list}

keys = set(arxiv_taxonomy.keys())

label_map = {
    "stat": "Statistics",
    "q-fin": "Quantitative Finance",
    "q-bio": "Quantitative Biology",
    "cs": "Computer Science",
    "math": "Mathematics"
} # anything else is physics

def filter_data_by_categories(subset, threshold=100):
    """
    Will filter a data frame with "categories" to only contain articles tagged as 
    computer science and will then relabel and add plaintext labels from the 
    arxiv taxonomy.
    """
    cats = subset["categories"]
    print("Filtering cs. categories...")
    cs_mask = ["cs." in cats for cats in tqdm(subset["categories"])]
    cs_subset = subset.loc[cs_mask]
    cats = cs_subset["categories"].to_list()
    cats = [cat.split(" ") for cat in cats]
    cs_mask = []
    print("Filtering by majority vote...")
    for item in tqdm(cats): 
        count = 0
        for label in item:
            if label.startswith("cs."):
                count += 1
        if count >= (len(item)/2):
            cs_mask.append(True)
        else:
            cs_mask.append(False)
    cs_subset_strict = cs_subset.loc[cs_mask]
    cs_l1_labels = cs_subset_strict["categories"].to_list()
    cs_l1_hardlabels = []
    for item in tqdm(cs_l1_labels):
        temp = item.split(" ")
        temp = list(filter(lambda a: a.startswith("cs."), temp))
        temp = " ".join(temp)
        cs_l1_hardlabels.append(temp)
    counter_map = Counter(cs_l1_hardlabels)
    hardlabel_counts = [counter_map[label] for label in cs_l1_hardlabels]
    cs_subset_strict["l1_labels"] = cs_l1_hardlabels
    cs_subset_strict["l1_counts"] = hardlabel_counts
    cs_keys = [key for key in arxiv_taxonomy.keys() if key.startswith("cs.")]
    cs_taxonomy = {key:arxiv_taxonomy[key] for key in cs_keys}
    cs_taxonomy["cs.LG"] = "Machine Learning"
    countmask = [count > threshold for count in hardlabel_counts]
    cs_subset_strict_filtered = cs_subset_strict.loc[countmask]
    filtered_l1_labels = cs_subset_strict_filtered["l1_labels"].to_list()
    def get_plaintext_name(l1_labels):
        p_labels = []
        for item in l1_labels:
            tmp = item.split(" ")
            plaintext_labels = [cs_taxonomy[k] for k in tmp]
            plaintext_labels = " & ".join(plaintext_labels)
            p_labels.append(plaintext_labels)
        return p_labels
    plaintext_labels = get_plaintext_name(filtered_l1_labels)
    cs_subset_strict_filtered["plaintext_labels"] = plaintext_labels
    return cs_subset_strict_filtered

# cats_2020 = subset_2020["categories"]
# # selecting *all* papers that contain "cs." in their categories
# cs_mask_2020 = ["cs." in cats for cats in tqdm(subset_2020["categories"])]
# cs_subset_2020 = subset_2020.loc[cs_mask_2020]
# # lets select only those papers where the majority of labels is "cs"
# # since we will only use these labels to calibrate the clustering performance of our models
# cats_2020 = cs_subset_2020["categories"].to_list()
# cats = [cat.split(" ") for cat in cats_2020]

# cs_mask = []
# for item in tqdm(cats): 
#     count = 0
#     for label in item:
#         if label.startswith("cs."):
#             count += 1
#     if count >= (len(item)/2):
#         cs_mask.append(True)
#     else:
#         cs_mask.append(False)

# cs_subset_2020_strict = cs_subset_2020.loc[cs_mask]
# 63533 papers remain
# lets convert labels to stricter categories: remove all non-cs labels and maybe group multiple categories
# cs_l1_labels = cs_subset_2020_strict["categories"].to_list()
# cs_l1_hardlabels = []
# for item in tqdm(cs_l1_labels):
#     temp = item.split(" ")
#     temp = list(filter(lambda a: a.startswith("cs."), temp))
#     temp = " ".join(temp)
#     cs_l1_hardlabels.append(temp)

# now we use threshold to filter for papers that are "easy" to classify (more than 1 paper
# of the same exact label exists)
# counter_map = Counter(cs_l1_hardlabels)
# hardlabel_counts = [counter_map[label] for label in cs_l1_hardlabels]

# cs_subset_2020_strict["l1_labels"] = cs_l1_hardlabels
# cs_subset_2020_strict["l1_counts"] = hardlabel_counts

# cs_keys = [key for key in arxiv_taxonomy.keys() if key.startswith("cs.")]
# cs_taxonomy = {key:arxiv_taxonomy[key] for key in cs_keys}
# cs_taxonomy["cs.LG"] = "Machine Learning"


# countmask = [count > 100 for count in hardlabel_counts]
# cs_subset_2020_strict_filtered = cs_subset_2020_strict.loc[countmask]
# filtered_l1_labels = cs_subset_2020_strict_filtered["l1_labels"].to_list()
# # check if all 40 L1 cs subcategories are present
# all(key in filtered_l1_labels for key in cs_keys)
# # not all keys are present as singlet categories => we expect less popular categories to 
# # occur paired with other categories
# len(Counter(filtered_l1_labels))
# # 77 distinct categories remain, let's see if this is workable
# # the computer science category has 40 distinct subcategories, we expect some papers tofall into 
# # a mix of these

# # lets construct a mapping to clarify category names: 

# def get_plaintext_name(l1_labels):
#     p_labels = []
#     for item in l1_labels:
#         tmp = item.split(" ")
#         plaintext_labels = [cs_taxonomy[k] for k in tmp]
#         plaintext_labels = " & ".join(plaintext_labels)
#         p_labels.append(plaintext_labels)
#     return p_labels

# plaintext_labels = get_plaintext_name(filtered_l1_labels)
# cs_subset_2020_strict_filtered["plaintext_labels"] = plaintext_labels

# to make sure the cluster evaluation metrics are sound we only keep documents that
# fall into classes which appear in every single one of the years respectively


cs_subset_2020_strict_filtered = filter_data_by_categories(subset_2020) 
cs_subset_2021_strict_filtered = filter_data_by_categories(subset_2021) 
cs_subset_2022_strict_filtered = filter_data_by_categories(subset_2022)

labelset_2020 = set(cs_subset_2020_strict_filtered["plaintext_labels"])
labelset_2021 = set(cs_subset_2021_strict_filtered["plaintext_labels"])
labelset_2022 = set(cs_subset_2022_strict_filtered["plaintext_labels"])

# remove labels that do not appear in all three years
labelset_total = labelset_2020.intersection(labelset_2021.intersection(labelset_2022))
# 64 classes appear in all three years
def filter_by_labelset(df, labelset=labelset_total):
    """
    Remove entries of a dataframe if their labels are not present in a predefined labelset
    """
    mask = [lab in labelset for lab in df["plaintext_labels"]]
    ret = df.loc[mask]
    return ret

cs_subset_2020_strict_filtered = filter_by_labelset(cs_subset_2020_strict_filtered)
cs_subset_2021_strict_filtered = filter_by_labelset(cs_subset_2021_strict_filtered)
cs_subset_2022_strict_filtered = filter_by_labelset(cs_subset_2022_strict_filtered)

assert len(Counter(cs_subset_2020_strict_filtered["plaintext_labels"])) <= len(labelset_total)
assert len(Counter(cs_subset_2021_strict_filtered["plaintext_labels"])) <= len(labelset_total)
assert len(Counter(cs_subset_2022_strict_filtered["plaintext_labels"])) <= len(labelset_total)
# the number of papers left over after the filtering process grows with each consecutive year

#### Supervised Topic Modeling
# Goal: Evaluate Clustering Performance given we already know the correct category labels
#   This allows us to set adequate hyperparameters

docs_2020 = cs_subset_2020_strict_filtered["abstract"].to_list()
# removing newline characters
docs_2020 = [doc.replace('\n', ' ').replace('\r', '') for doc in docs_2020]
labels_2020 = cs_subset_2020_strict_filtered["plaintext_labels"]
# converting labels to numeric
plaintext_map = {k:v for k, v in zip(set(labels_2020), list(range(0, len(set(labels_2020)), 1)))}
numeric_map_2020 = {v:k for k, v in plaintext_map.items()}
numeric_labels_2020 = [plaintext_map[k] for k in labels_2020]

timestamps_2020 = cs_subset_2020_strict_filtered["v1_datetime"].to_list()

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

# Skip over dimensionality reduction, replace cluster model with classifier,
# and reduce frequent words while we are at it.
empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Create a fully supervised BERTopic instance
topic_model= BERTopic(
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf,
        ctfidf_model=ctfidf_model,
        verbose=True
)
# embedding takes about 40 seconds for the full year
topics, probs = topic_model.fit_transform(docs_2020, y=numeric_labels)

topic_info = topic_model.get_topic_info()
# we obtain a naive classifier that simply extracted
# topic representations for all our documents 
# depending on their prior classes
mappings = topic_model.topic_mapper_.get_mappings()
mappings = {value: numeric_map[key] for key, value in mappings.items()}

topic_info["Class"] = topic_info.Topic.map(mappings)

#### manual topic modeling
# idea: fit manual model first then perform online modeling as
# downstream task

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN

# Prepare our empty sub-models and reduce frequent words while we are at it.
# precompute embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model =  UMAP(n_neighbors=10, n_components=10, metric='cosine', low_memory=False, random_state=42)
empty_dimensionality_model = BaseDimensionalityReduction()
empty_cluster_model = BaseCluster()
cluster_model = HDBSCAN(min_cluster_size=100, metric='euclidean', prediction_data=True)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Fit BERTopic without actually performing any clustering
topic_model= BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        ctfidf_model=ctfidf_model,
        verbose=True,
        nr_topics = 65,
)
# supervised fit on 2020 docs
embeddings_2020 = embedding_model.encode(docs_2020, show_progress_bar=True)
topics, probs = topic_model.fit_transform(docs_2020, embeddings_2020, y=numeric_labels_2020)
topic_info = topic_model.get_topic_info()

mappings = topic_model.topic_mapper_.get_mappings()
mappings = {value: numeric_map_2020[key] for key, value in mappings.items()}

topic_info["Class"] = topic_info.Topic.map(mappings)

# now perform dynamic modeling as downstream task
topics_over_time = topic_model.topics_over_time(docs_2020, timestamps_2020, nr_bins=52)
# the frequencies add up to the respective document count
# we obtain topic frequency counts for every timestamp
# but we do have labels now, are all of the entries correct? 
# this is where we need to evaluate clustering
# then we evaluate changes over time to detect trends
# then we evaluate coherence & diversity

tovert = topic_model.visualize_topics_over_time(topics_over_time, topics=[0, 1, 2, 3, 4])
tovert.write_image(str(ARTIFACTS) + "/tovert_2020.png")

# lets look at assigned labels and true labels, maybe we need to test this with 2021 data
docinfo = topic_model.get_document_info(docs_2020)
classes = docinfo["Topic"]
classes = [mappings[cls] for cls in classes]
docinfo["Class"] = classes
all(cls == lab for cls, lab in zip(classes, labels_2020))
# all classes are assigned correctly, we need to see if we can 

### predict labels of the 2021 data correctly with this
plaintext_map_2021 = {k:v for k, v in zip(set(labels_2021), list(range(0, len(set(labels_2021)), 1)))}
numeric_map_2021 = {v:k for k, v in plaintext_map_2021.items()}
numeric_labels_2021 = [plaintext_map_2021[k] for k in labels_2021]
docs_2021 = cs_subset_2021_strict_filtered["abstract"].to_list()
docs_2021 = [doc.replace('\n', ' ').replace('\r', '') for doc in docs_2021]
labels_2021 = cs_subset_2021_strict_filtered["plaintext_labels"]
numeric_labels_2021 = numeric_labels = [plaintext_map_2021[k] for k in labels_2021]

# TODO: only keep documents in both years that exist in classes that exist in both
# years respectively to clean up cluster evaluation procedures

timestamps_2021 = cs_subset_2021_strict_filtered["v1_datetime"].to_list()

#### Predict New instances from 2021
from sentence_transformers import SentenceTransformer
from umap import UMAP

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs_2021, show_progress_bar=True)

##Set the random state in the UMAP model to prevent stochastic behavior
umap_model = UMAP(min_dist=0.0, metric='cosine', random_state=42)

pred_topics, pred_probs = topic_model.transform(docs_2021, embeddings)
pred_plaintext = [mappings[t] for t in pred_topics]
labels_2021[0:10]

# lets create a strict confusion matrix and compute silhouette scores etc

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_2021, pred_plaintext)
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.figure(figsize = (50,50))
sns.heatmap(cm, annot=True, fmt="d", linewidth=.5)
plt.savefig(str(ARTIFACTS) + "/cm_2021_10_comp_10_neighbors.png")

# idea: use semi supervised learning with growing number of -1 labels and determine 
# cutoff point for label accuracy

umap_embeddings = topic_model.umap_model.transform(embeddings)
indices = [index for index, topic in enumerate(pred_topics) if topic != -1]
X = umap_embeddings[np.array(indices)]
labels = [topic for index, topic in enumerate(pred_topics) if topic != -1]
from sklearn.metrics import silhouette_score
silhouette_score(X, labels)
# -0.0926 for default umap with 15 neighbors and 5 components
# accuracy scores
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos

precision = np.sum(true_pos / (true_pos + false_pos)) # 23.67
recall = np.sum(true_pos / (true_pos + false_neg)) # 31.41
accuracy = np.sum(true_pos / (np.sum(cm))) # 0.36

np.argmax(false_neg), np.argmax(false_pos)
mappings[12], mappings[16]
Counter(cs_subset_2020_strict_filtered["plaintext_labels"])["Data Structures and Algorithms"]
Counter(cs_subset_2020_strict_filtered["plaintext_labels"])["Distributed, Parallel, and Cluster Computing"]




#### The supervised model is basically useless in terms of evaluating clustering performance
# since nothing has an impact on the predicted classes. 
# Here we try a semi supervised hierarchical model: 
# Goal: 
#   Fit model that yields ~100 classes
#   Then merge classes to target labels

import numpy as np
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
docs_2020[0:10], labels_2020[0:10], numeric_labels_2020[0:10]

embeddings_2020 = sentence_model.encode(docs_2020)
# TODO: benchmark different embedding updating methods (embetter etc.)
# Average embeddings based on the class
embeddings_averaged = embeddings_2020.copy()
for target in set(numeric_labels_2020):
    indices = [index for index, t in enumerate(numeric_labels_2020) if t==target]
    embeddings_averaged[indices] = np.average([embeddings_averaged[indices], np.mean(embeddings_averaged[indices])], weights=[1, 1])

len(embeddings_averaged)

from umap import UMAP
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Sub models
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
cluster_model = KMeans(n_clusters=150) # replacing HDBSCAN since we do not wish to obtain outliers
vectorizer_model = CountVectorizer(stop_words="english")

# Train our semi-supervised topic model
topic_model = BERTopic(hdbscan_model=cluster_model, umap_model=umap_model, vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(docs_2020, embeddings_averaged, y=numeric_labels_2020)
topic_info = topic_model.get_topic_info()
# 150 topics like specified in kmeans
import pandas as pd
from tqdm import tqdm
hierarchical_topics = topic_model.hierarchical_topics(docs_2020)
# find matches between original labels and topics in the hierarchy
results = pd.DataFrame(columns = ["Parent_ID", "Topics", "Target", "Overlap1", "Overlap2", "Score", "Score1"])
for parent_id, topic_set in tqdm(zip(hierarchical_topics.Parent_ID.values, hierarchical_topics.Topics.values)):
    hierarchy_indices = {index for topic in topic_set for index, val in enumerate(topics) if val == topic}
    for target in set(numeric_labels_2020):
        y_indices = set([index for index, t in enumerate(numeric_labels_2020) if t == target])
        overlap1 = round(len(hierarchy_indices.intersection(y_indices)) / len(y_indices) * 100, 1)
        overlap2 = round(len(hierarchy_indices.intersection(y_indices)) / len(hierarchy_indices) * 100, 1)
        score = round(np.mean((overlap1, overlap2)), 1)
        score1 = round(min(overlap1, overlap2), 1)
        results.loc[len(results), :] = [parent_id, topic_set, target, overlap1, overlap2, score, score1]
# add unmerged topics
for parent_id, topic_set in tqdm(zip(set(topics), [[topic] for topic in set(topics)])):
    hierarchy_indices = {index for topic in topic_set for index, val in enumerate(topics) if val==topic}
    for target in set(numeric_labels_2020):
        y_indices = set([index for index, t in enumerate(numeric_labels_2020)])
        overlap1 = round(len(hierarchy_indices.intersection(y_indices)) / len(y_indices) * 100, 1)
        overlap2 = round(len(hierarchy_indices.intersection(y_indices)) / len(hierarchy_indices) * 100, 1)
        score = round(np.mean((overlap1, overlap2)), 1)
        score1 = round(min(overlap1, overlap2), 1)
        results.loc[len(results), :] = [parent_id, topic_set, target, overlap1, overlap2, score, score1]
results = results[results['Score']==results.groupby('Target')['Score'].transform('max')]

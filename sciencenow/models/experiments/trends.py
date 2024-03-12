"""
DEPRECATED
"""

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

cs_subset_2020_strict_filtered = filter_data_by_categories(subset_2020) 
cs_subset_2021_strict_filtered = filter_data_by_categories(subset_2021) 
cs_subset_2022_strict_filtered = filter_data_by_categories(subset_2022)

# obtaining labelsets is part of supervised modeling 
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

with open (str(ARTIFACTS) + "/cs_2020.pickle", "wb") as handle:
    pickle.dump(docs_2020, handle, protocol=pickle.HIGHEST_PROTOCOL)

labels_2020 = cs_subset_2020_strict_filtered["plaintext_labels"]
# converting labels to numeric
plaintext_map = {k:v for k, v in zip(set(labels_2020), list(range(0, len(set(labels_2020)), 1)))}
numeric_map_2020 = {v:k for k, v in plaintext_map.items()}
numeric_labels_2020 = [plaintext_map[k] for k in labels_2020]

with open(custom_path / "labels_2020.pkl", "wb") as file:
    pickle.dump(numeric_labels_2020, file)


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
from umap import UMAP
from sentence_transformers import SentenceTransformer

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
        nr_topics = 80,
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
cluster_model = KMeans(n_clusters=85) # replacing HDBSCAN since we do not wish to obtain outliers
vectorizer_model = CountVectorizer(stop_words="english")

# Train our semi-supervised topic model
topic_model = BERTopic(hdbscan_model=cluster_model, umap_model=umap_model, vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(docs_2020, embeddings_averaged, y=numeric_labels_2020)
topic_info = topic_model.get_topic_info()
# number of topics specified in kmeans
import pandas as pd
from tqdm import tqdm
hierarchical_topics = topic_model.hierarchical_topics(docs_2020)
# let's save the hierarchy first and join classes based on overlapping heuristics
hier_viz = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
hier_viz.write_image(str(ARTIFACTS) + "/cs_hierarchy_85kmeans.png")


# find matches between original labels and topics in the hierarchy
results = pd.DataFrame(columns = ["Parent_ID", "Topics", "Target", "Overlap1", "Overlap2", "Score", "Score1"])
for parent_id, topic_set in tqdm(zip(hierarchical_topics.Parent_ID.values, hierarchical_topics.Topics.values)):
    # hierarchy_indices => indices of papers that fall in either class of the topic_set (grows with size of topic_set)
    hierarchy_indices = {index for topic in topic_set for index, val in enumerate(topics) if val == topic}
    # now loop over all numeric label classes 
    # we are trying to find which class of original labels overlaps most with the target set in our 
    # hierarchy. Ex: Topics 53 & 29 are both related to reinforcement learning, 
    # can we find a label class that matches this?
    # There is no label "Reinforcement Learning" in our Label set.
    for target in set(numeric_labels_2020):
        # indices of papers that match target 0:63
        y_indices = set([index for index, t in enumerate(numeric_labels_2020) if t == target])
        # % of papers that belong to topic set that belong to target label 0:63
        overlap1 = round(len(hierarchy_indices.intersection(y_indices)) / len(y_indices) * 100, 1)
        # % of topic set that belongs to target label 0:63
        overlap2 = round(len(hierarchy_indices.intersection(y_indices)) / len(hierarchy_indices) * 100, 1)
        score = round(np.mean((overlap1, overlap2)), 1)
        score1 = round(min(overlap1, overlap2), 1)
        results.loc[len(results), :] = [parent_id, topic_set, target, overlap1, overlap2, score, score1]

results_RL = results
results_RL = results_RL.sort_values(["Overlap1", "Overlap2"], ascending=[False,False])
# RL papers fit best into classes 1, 51, and 56
# our original labeling scheme is too broad to compare topic clusters meaningfully.


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

test = results.drop_duplicates(subset="Topics")

# group 59 is about graphs and graph_vertices etc. 
# target should also be 59, let's see if this is correct. 
indices = [index for index, t in enumerate(numeric_labels_2020) if t==56]
rl_papers = [docs_2020[i] for i in indices]
rl_labels = [labels_2020.to_list()[i] for i in indices]
# they have all been correctly identified as belonging to the same class, 
# but we don't find a match from topic to this class, they are not graph related
# they belong to Robotics & Computer Vision and Pattern Recognition
graph_papers[0]
len(labels_2020), len(numeric_labels_2020)


# these results are example groupings based on maximum overlap
# we need a heuristic to select meaningful topic groups that are unique

# relabeling classes based on these merging results
topic_info
docinfo = topic_model.get_document_info(docs_2020)
classes = docinfo["Topic"]
# if results["Topics"] is not [0] then the class 
# needs to be relabeled with numeric_map_2020

topic_groups = results["Topics"]
topic_targets = results["Target"]
# remove first two lists since they define the roots of the hierarchy
topic_group_list_of_lists = topic_groups.to_list()[1:]

def find_list_index(list_of_lists, number):
    for i, sublist in enumerate(list_of_lists):
        if number in sublist:
            return i  # Return the index if the number is found in the sublist
    return -1  # Return -1 if the number is not found in any sublist

indices = []
for cls in tqdm(classes):
    indices.append(find_list_index(topic_group_list_of_lists, number=cls))

merged_labels = []
for cls, ind in tqdm(zip(classes, indices)):
    if ind == -1: # if class was not part of remapping list we keep original mapping
        merged_labels.append(cls)
    else: 
        merged_labels.append(topic_targets.to_list()[ind])

Counter(merged_labels)
len(Counter(merged_labels))

# we need to merge "backwards" 
print(results)
# if cls == 62: target is 62
# if cls is contained in any of the longer

# we need to merge in reverse: 
# first relabel based on singlets: 
# 62 remains, 61 remains, 59 remains etc
# then relabel based on doubles: 
# [1, 51] => 45, [7, 68] => 16 etc.
# then relabel based on triples: 
# [1, 51, 88] => 7
# does this make sense??
topic_info["Representation"][1]
topic_info["Representation"][51]
topic_info["Representation"][88]

# these topics would end up being grouped as
numeric_map_2020[7]
# computational Geometry
# in the earlier step as
numeric_map[45]
# this seems correct.
# so we only relabel those that have not yet appeared at a deeper level
# in the hierarchy
# [1, 51] -> 45 (Data Structures and Algorithms & Discrete Mathematics)
# [88] -> Computational Geometry
numeric_map[60]
# [92] -> Systems and Control
# we repeat this until we have only our 64 target classes remaining

def merge_topics(result_table, original_classes):
    """
    Perform Backwards Merging to recover target hierarchy.
    """
    # only use mappings with the largest overlap scores
    uniques = result_table.drop_duplicates(subset="Topics")
    # for singlets/unmerged clusters no remapping needs to be done
    singlets = result_table[result_table['Topics'].apply(lambda x: x == [0])]
    # find the deepest level in hierarchy where original class is found
    def find_last_index_with_number(list_of_lists, target_number):
        for i, sublist in enumerate(reversed(list_of_lists)):
            if target_number in sublist:
                return len(list_of_lists) - 1 - i
        return None  # Return None if the target number is not found in any sublist
    # step through hierarchy backwards
    merged_labels = []
    for cls in original_classes.to_list(): # original_classes
        if cls in singlets["Target"].to_list():
            merged_labels.append(cls)
            continue
        index = find_last_index_with_number(uniques["Topics"].to_list(), cls)
        if index is not None:
            merged_labels.append(uniques["Target"].to_list()[index])
        else:
            merged_labels.append(uniques["Target"].to_list()[0])
    return merged_labels

merged_labels = merge_topics(results, original_classes=docinfo["Topic"])
idx = find_last_index_with_number(results["Topics"].to_list(), 44)
results["Target"].to_list()[idx]
len(set(merged_labels))

## Question: How to deal with identical cluster pairs that map to different 
# Targets?
topic_info["Representation"][44]
topic_info["Representation"][3]
numeric_map_2020[41]
numeric_map_2020[61]
# The Labels & Representations match but doing a 50/50 split will not work.
# Calculate bidiractional overlaps to resolve these cases
# Calculate cluster Distances with embeddings => Find out Which Target 
# Matches best
# Other approach: Loop over Targets first and then fill in the remaining Topics
# Qualitative validation with expert feedback for trend detection
# Quantitative via expected and predicted counts over time (?)
# look into citation count prediction
# Arxiv Labels may need to be grouped a priori, some splits may not make sense



# targets 2, 3, 4, 9 are singlets
topic_info["Representation"][0]
numeric_map_2020[2]
numeric_map_2020[3]
numeric_map_2020[4]
numeric_map_2020[9]
numeric_map_2020[44]

# some values only appear in the largest topic cluster => should also be treated as singlets?
def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))

missing = missing_elements(list(set(merged_labels)))
# numeric_map_2020[41] # Computers and Society
# topic_info["Representation"][28] # 'students', 'covid19', 'pandemic', 'health', 'mobility'
# numeric_map_2020[34] # Computation and Language 
# topic_info["Representation"][30] # 'entity', 'text', 'language', 'event', 'models', 'nlp'
# numeric_map_2020[44] # Computer Vision and Pattern Recognition
# topic_info["Representation"][38] # 'image', 'object', 'classification', 'cnns', 'training', 'convolutional',
# numeric_map_2020[45] # Data Structures and Algorithms
# topic_info["Representation"][51] # graphs, graph, vertices, number, vertex, 
# numeric_map_2020[5] # Computer Vision and Pattern Recognition & Machine Learning
# topic_info["Representation"][50] # federated, f1, distributed, communication,
# numeric_map_2020[44] # Computer Vision and Pattern Recognition
# topic_info["Representation"][54] # 'plant', 'images', 'satellite', 'imagery', 'species', 'classification',
# numeric_map_2020[46] # Networking and Internet Architecture
# topic_info["Representation"][55] # 'network', 'traffic', 'service', 'routing', 'sdn', 'packet', 'internet',
# numeric_map_2020[44] # Computer Vision and Pattern Recognition
# topic_info["Representation"][56] # Captioning, visual, video, vqa

# 30 appears in hierarchy, should be mapped to 34
# doublecheck all missing values
index = find_last_index_with_number(uniques["Topics"].to_list(), 56)
uniques["Target"].to_list()[index] # works

numeric_map[55]
[28, 30, 38, 50, 51, 54, 55, 56]
test = result_table[result_table['Target'].apply(lambda x: x == 30)]

# we cannot filter by unique else we lose information about targets
# we need to preserve target structure to have a 64 label mapping

tree = topic_model.get_topic_tree(hierarchical_topics)
print(tree)

# compare merged_labels to numeric_labels_2020
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(numeric_labels_2020, merged_labels)
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.figure(figsize = (50,50))
sns.heatmap(cm, annot=True, fmt="d", linewidth=.5)
plt.savefig(str(ARTIFACTS) + "/cm_2020_200_kmeans.png")

# -0.0926 for default umap with 15 neighbors and 5 components
# accuracy scores
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos

precision = np.sum(true_pos / (true_pos + false_pos)) # 23.67
recall = np.sum(true_pos / (true_pos + false_neg)) # 31.41
accuracy = np.sum(true_pos / (np.sum(cm))) # 0.36
"""
Results: 150kmeans:
    Precision: 0.28
    Recall: 0.28
    Accuracy: 0.28
"""

# TODO: Eval Models that produce outliers#
# TODO: Fit Level 2 Model to Outlier Class

"""
DEPRECATED
"""

# semi supervised hierarchical model
# arxiv papers fall in a natural hierarchy
# larger topics like physics, maths, computer science, etc. are split into subtopics
# can we leverage this structure for topic modeling?

import pandas as pd
import numpy as np
from bertopic import BERTopic
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/")
ROOT.exists()
ARXIV_PATH = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/arxivdata/arxiv-metadata-oai-snapshot.json")
ARXIV_PATH.exists()
ARTIFACTS = ROOT / Path("artifacts/model2M")


def setup(n=10000):
    with open(ARTIFACTS / "sentence_list_full.pkl", "rb") as file:
        sentence_list_full = pickle.load(file)
    with open(ARTIFACTS / "timestamp_list_full.pkl", "rb") as file:
        timestamp_list_full = pickle.load(file)
    embeddings = np.load(ARTIFACTS / "embeddings2M.npy")
    with open(ARTIFACTS / "vocab2m.pkl", "rb") as file:
        vocab = pickle.load(file)
    vocab_reduced = [word for word, frequency in vocab.items() if frequency >= 50]
    reduced_embeddings = np.load(ARTIFACTS / "reduced_embeddings2M.npy")
    reduced_embeddings_2d = np.load(ARTIFACTS / "reduced_embeddings2M_2d.npy")
    clusters = np.load(ARTIFACTS / "hdb_clusters2M.npy")
    return (
        sentence_list_full[0:n],
        timestamp_list_full[0:n],
        embeddings[0:n],
        vocab_reduced[0:n],
        reduced_embeddings[0:n],
        reduced_embeddings_2d[0:n],
        clusters[0:n],
    )


(
    docs,
    timestamps,
    embeddings,
    vocab,
    reduced_embeddings,
    reduced_embeddings_2d,
    clusters,
) = setup(n=10000)


from sentence_transformers import SentenceTransformer

# sample obtained from evaluation with label mapping
# TODO: wrap in clean preprocessing file
# dataframe with 10k samples
len(sample)
len(sample_categories)

label_to_numeric = {
    "Outlier": -1,    
    "Physics": 0, 
    "Mathematics": 1,
    "Computer Science": 2, 
    "Quantitative Biology": 3, 
    "Quantitative Finance": 4,
    "Statistics":5,
}

numeric_to_label = {
    -1: "Outlier",
    0: "Physics", 
    1: "Mathematics",
    2: "Computer Science", 
    3: "Quantitative Biology", 
    4: "Quantitative Finance",
    5: "Statistics",
}

docs = sample["abstract"].tolist()
target_names = sample_categories
y = np.array([label_to_numeric[target] for target in target_names])

# prepare embeddings: 
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# Semi Supervised Topic Modelling
# TODO: fine-tune embedding model using contrastive learning.
# Semi Supervised Topic Modeling will only nudge the dimensionality reductino process 
# toward the given labels. 
# For a Hierarchical topic model this has both advantages and disadvantages. 
# Advantage: We uncover more than the base classes
# Disadvantage: We need to control by how much the subclasses are separated
# we want separation to obtain subclasses but not so much that it breaks the 
# super classes in the hierarchy
# this is why we fine tune the embeddings using embetter
from embetter.finetune import ForwardFinetuner
tuner = ForwardFinetuner(n_epochs=500, learning_rate=0.01, hidden_dim=300)

tuner.fit(embeddings, y)

embetter_embeddings = tuner.transform(embeddings)

# visualizing the embeddings in 2D:
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
embedding_sets = [(embeddings, "Embeddings"), (embetter_embeddings, "Embetter Embeddings")]
fig, ax = plt.subplots(1, len(embedding_sets), figsize = (60, 30))

# reduce dimensionality and visualize:
for index, (embedding_set, title) in enumerate(embedding_sets):
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, metric="cosine", random_state=42).fit_transform(embedding_set, y=y)
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=y, ax=ax[index], alpha=0.3, size=1, palette=sns.color_palette("tab20"))
    ax[index].set_title(title, fontdict={"fontsize": 50})


fig.savefig("STransformer vs Embetter")

# it seems that the separation into distinct clusters that match with their
# respective labels was a lot better before finetuning
# but the cluster homogenity was a lot worse
# lets see what kind of effect this has on a hierarchical model

## Hierarchical Model
# We cannot be sure that the found subtopics will be merged back into the 
# supertopics they belong to. 
# => Need a way to match merged topics with the initil labels
# => For each merged topic we count how many documents overlap with the 
# documents for a specific label

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Sub models
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
# replacing HDBSCAN with KMeans since our labeled dataset does not contain outliers
cluster_model = KMeans(n_clusters=176) # the arxiv taxonomy had 176 Level 1 Topics
vectorizer_model = CountVectorizer(stop_words="english")
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Train our semi-supervised topic model
topic_model_default = BERTopic(hdbscan_model=cluster_model, umap_model=umap_model, vectorizer_model=vectorizer_model)
topics_default, probs_default = topic_model_default.fit_transform(docs, embeddings, y=y)

# train on fine tuned embeddings
topic_model_finetuned = BERTopic(hdbscan_model=cluster_model, umap_model=umap_model, vectorizer_model=vectorizer_model)
topics_finetuned, probs_finetuned = topic_model_finetuned.fit_transform(docs, embetter_embeddings, y=y)

# train on default embeddings with hdbscan clustering
topic_model_hdbscan = BERTopic(hdbscan_model=hdbscan_model, umap_model=umap_model, vectorizer_model=vectorizer_model)
topics_hdbscan, probs_hdbscan = topic_model_hdbscan.fit_transform(docs, embeddings, y=y)


# matching (merged) topics that belong to the initial labels: 
hierarchical_topics_default = topic_model_default.hierarchical_topics(docs)
hierarchical_topics_finetuned = topic_model_finetuned.hierarchical_topics(docs)
hierarchical_topics_hdbscan = topic_model_hdbscan.hierarchical_topics(docs)

def tabulate_results(topics, hr_topics, y=y):
    results = pd.DataFrame(columns = ["Parent_ID", "Topics", "Target", "Overlap1", "Overlap2", "Score", "Score1"])
    for parent_id, topic_set in tqdm(zip(hr_topics.Parent_ID.values, hr_topics.Topics.values)):
        hierarchy_indices = {index for topic in topic_set for index, val in enumerate(topics) if val == topic}
        for target in set(y):
            y_indices = set([index for index, t in enumerate(y) if t == target])
            # how much do hierarchy indices that match the target overlap with the target indices
            overlap1 = round(len(hierarchy_indices.intersection(y_indices)) / len(y_indices) * 100, 1)
            # what percentage of hierarchy indices matches the target 
            overlap2 = round(len(hierarchy_indices.intersection(y_indices)) / len(hierarchy_indices) * 100, 1)
            score = round(np.mean((overlap1, overlap2)), 1) # mean overlap
            score1 = round(min(overlap1, overlap2), 1) # minimum overlap
            results.loc[len(results), :] = [parent_id, topic_set, target, overlap1, overlap2, score, score1]
    # Add unmerged topics
    for parent_id, topic_set in tqdm(zip(set(topics), [[topic] for topic in set(topics)])):
        hierarchy_indices = {index for topic in topic_set for index, val in enumerate(topics) if val == topic}
        for target in set(y):
            y_indices = set([index for index, t in enumerate(y) if t == target])
            overlap1 = round(len(hierarchy_indices.intersection(y_indices)) / len(y_indices) * 100, 1)
            overlap2 = round(len(hierarchy_indices.intersection(y_indices)) / len(hierarchy_indices) * 100, 1)
            score = round(np.mean((overlap1, overlap2)), 1)
            score1 = round(min(overlap1, overlap2), 1)
            results.loc[len(results), :] = [parent_id, topic_set, target, overlap1, overlap2, score, score1]    
    results = results[results['Score']==results.groupby('Target')['Score'].transform('max')]
    return results

default_results = tabulate_results(topics=topics_default, hr_topics=hierarchical_topics_default)
finetuned_results = tabulate_results(topics=topics_finetuned, hr_topics=hierarchical_topics_finetuned)
hdbscan_results = tabulate_results(topics=topics_hdbscan, hr_topics=hierarchical_topics_hdbscan)

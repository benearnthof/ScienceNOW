#### Bootstrap Topic Model
# Fit Model to data to derive first layer topics
# then fit partial model on clusters to derive more detailed topics

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
import time
from embetter.text import SentenceEncoder
import pandas as pd

docs = docs_2020
labs = labels_2020.to_list() # not needed since they are not informative enough for meaningful cluster comparison
n_labs = numeric_labels_2020

# Idea: Fit Unsupervised Model to data first to obtain clusters and labels
# Step 0: Gridsearch to compare impact of hyperparameters on the number of cluster and outliers that are 
# generated

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# precompute embeddings
embeddings_2020 = embedding_model.encode(docs_2020, show_progress_bar=True)

# neighbors: 5, 10 , 15, 20
# n_components: 2:20
# min_cluster_size: 10, 20, 30, ..., 250
# TODO: evaluate scores => obtain corpus and vocab needed for octis etc. 
nbors = range(10, 25, 1)
# for anything more than 10 components the clustering saturates at around 65 clusters
# 5 neighbors and min cluster size of 100
ncomp = range(3, 12, 1) 
mclust = range(10, 250, 10)

len(nbors) * len(ncomp)

# 8640 * 50 / 3600 120 hours => need smaller gridsearch
results = pd.DataFrame(columns = ["neighbors", "components", "topics", "outliers"])
for n in nbors:
    for c in ncomp:
        umap_model =  UMAP(n_neighbors=n, n_components=c, metric='cosine', low_memory=False, random_state=42)
        cluster_model = HDBSCAN(min_cluster_size=100, metric='euclidean', prediction_data=True)
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        # Fit BERTopic without actually performing any clustering
        topic_model= BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=cluster_model,
                ctfidf_model=ctfidf_model,
                verbose=False,
        )
        topics, _ = topic_model.fit_transform(docs, embeddings_2020)
        t = len(set(topics))
        topic_info = topic_model.get_topic_info()
        # num outliers
        o = len([x for x in topics if x == -1])
        results.loc[len(results), :] = [n, c, t, o]
        print(f"Neighbors: {n}, Components: {c}, Topics: {t}, Outliers: {o}")

# TODO: obtain proper preprocessing script
# TODO: clean up codebase
# TODO: compute performance scores => OCTIS trainer.py
# TODO: how investigate outliers
# TODO: add secondary & tertiary models

# after we optimize hyperparameters for coherence & diversity 
# we do a semi supervised fit with our newly obtained partial labels
# -1 for every document that is an outlier
# we repeat this until the document assignment no longer changes
# and or coherence and diversity no longer improve
# then we look at results and also begin fitting other models for outlier classes


















n_train = 200

# trying out embetter
import random
random.seed(42)
df_train = random.sample(docs, round(len(docs)*0.8))

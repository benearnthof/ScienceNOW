"""
DEPRECATED
"""

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
from pathlib import Path

#docs = docs_2020 # TODO: Clean up all preprocessing utils
#labs = labels_2020.to_list() # not needed since they are not informative enough for meaningful cluster comparison
#n_labs = numeric_labels_2020

# Idea: Fit Unsupervised Model to data first to obtain clusters and labels
# Step 0: Gridsearch to compare impact of hyperparameters on the number of cluster and outliers that are 
# generated

# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# precompute embeddings
# embeddings_2020 = embedding_model.encode(docs_2020, show_progress_bar=True)

# neighbors: 5, 10 , 15, 20
# n_components: 2:20
# min_cluster_size: 10, 20, 30, ..., 250
# TODO: evaluate scores => obtain corpus and vocab needed for octis etc. 
# nbors = range(10, 20, 2)
# for anything more than 10 components the clustering saturates at around 65 clusters
# 5 neighbors and min cluster size of 100
# ncomp = range(5, 11, 2) 
# mclust = range(25, 300, 25)

# len(nbors) * len(ncomp) * len(mclust) * 50 / 3600
from bertopic import BERTopic

# params = {
#     "verbose": True,
#     "umap_model": UMAP(n_neighbors = 15, n_components=5, metric='cosine', low_memory=False, random_state=42),
#     "hdbscan_model": HDBSCAN(min_cluster_size=1000, metric='euclidean', prediction_data=True),
#     "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True)
# }


# tm = BERTopic(**params)

from sentence_transformers import SentenceTransformer
from ScienceNOW.sciencenow.models.trainer import Trainer
from ScienceNOW.sciencenow.models.dataloader import DataLoader
from sklearn.feature_extraction.text import CountVectorizer

dataset, custom = "arxiv", True
data_loader = DataLoader(dataset)
data, timestamps = data_loader.load_docs()
#data, timestamps = docs, timestamps
# timestamps = [str(stamp) for stamp in timestamps]
# model = SentenceTransformer("all-mpnet-base-v2")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
#_, timestamps = data_loader.load_docs()
#

embeddings = sentence_model.encode(data, show_progress_bar=True)

# precompute reduced embeddings
class Dimensionality:
    """ Use this for pre-calculated reduced embeddings """
    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings
    def fit(self, X):
        return self
    def transform(self, X):
        return self.reduced_embeddings

from umap import UMAP
umap_model = UMAP(n_neighbors = 15, n_components=5, metric='cosine', low_memory=False, random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)


# topics, _ = tm.fit_transform(data, embeddings)

custom_path = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/artifacts/bootstrap")

# Extract vocab to be used in BERTopic
from collections import Counter
docs = data
vocab = Counter()
tokenizer = CountVectorizer().build_tokenizer()
for doc in tqdm(docs):
     vocab.update(tokenizer(doc))

with open(custom_path / "vocab.txt", "w") as file:
    for item in vocab:
        file.write(item+"\n")
file.close()

docs = [doc.replace("\n", " ") for doc in docs]
assert all("\n" not in doc for doc in docs)

with open(custom_path / "corpus.tsv", "w") as file:
    for document in docs:
        file.write(document + "\n")
file.close()

# now run every set of parameters three times to help combat randomness introduced by UMAP
# we analyze the influence of the number of components on the result aswell since ideally
# we would like to precompute the reduced embeddings for faster caching and performance
from tqdm import tqdm
results = []
len(range(1,50,5)) * len(range(25, 325, 25)) * 60 / 3600
for clust in tqdm(range(25, 525, 25)):
    for samp in tqdm(range(1, 25, 5)): # min_samples has a negligible impact on diversity and npmi
        params = {
        "verbose": False,
        "umap_model": Dimensionality(reduced_embeddings),
        "hdbscan_model": HDBSCAN(min_cluster_size=clust, min_samples=samp, metric='euclidean', prediction_data=True),
        "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True)
        }
        trainer = Trainer(dataset=dataset,
                        model_name="BERTopic",
                        params=params,
                        bt_embeddings=embeddings,
                        custom_dataset=custom,
                        bt_timestamps=None,
                        topk=5,
                        bt_nr_bins=10,
                        verbose=True)
        res = trainer.train()
        results.append(res)

#import pickle

with open(custom_path / "results.txt", "w") as file:
    file.write("\n".join(str(item) for item in results))
file.close()

with open(custom_path / "results.pkl", "rb") as file: 
    results = pickle.load(file)
file.close()

tops = [x[0]["Topics"] for x in results]
from collections import Counter

cnt = [Counter(x) for x in tops]
n_outs = [x[-1] for x in cnt]

minouts = np.argmin(n_outs)
maxouts = np.argmax(n_outs)
results[8][0]["Params"] # min_cluster_size=25, min_samples=41
results[8][0]["Scores"] # {'npmi': 0.08112440690032077, 'diversity': 0.7185185185185186}
Counter(results[8][0]["Topics"]) # 160 topics, 21194 outliers
results[80][0]["Params"] # min_cluster_size=225, min_samples=1
results[80][0]["Scores"] # {'npmi': 0.08216433282418788, 'diversity': 0.8588235294117647}
Counter(results[80][0]["Topics"]) # 50 topics, 12464 outliers

# rerunning evaluation with semi supervised model

from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
assert len(data) == len(embeddings) == len(numeric_labels_2020)

# Average embeddings based on the class
embeddings_averaged = embeddings.copy()
for target in set(numeric_labels_2020):
    indices = [index for index, t in enumerate(numeric_labels_2020) if t==target]
    embeddings_averaged[indices] = np.average([embeddings_averaged[indices], np.mean(embeddings_averaged[indices])], weights=[1, 1])

len(embeddings_averaged)

from umap import UMAP
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Train our semi-supervised topic model
# topic_model = BERTopic(hdbscan_model=cluster_model, umap_model=umap_model, vectorizer_model=vectorizer_model)
# topics, probs = topic_model.fit_transform(docs_2020, embeddings_averaged, y=numeric_labels_2020)

with open(custom_path / "labels_2020.pkl", "wb") as file:
    pickle.dump(numeric_labels_2020, file)

with open(custom_path / "labels_2020.pkl", "rb") as file:
    numeric_labels_2020 = pickle.load(file)


results = []
len(range(5,200,1)) * 60 / 3600
for clust in tqdm(range(5, 200, 1)):
    # min_samples has a negligible impact on diversity and npmi
    params = {
    "verbose": False,
    "umap_model": UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
    "hdbscan_model": KMeans(n_clusters=clust),
    "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True),
    "vectorizer_model": CountVectorizer(stop_words="english")
    }
    trainer = Trainer(dataset=dataset,
                    model_name="BERTopic",
                    params=params,
                    bt_embeddings=embeddings_averaged,
                    custom_dataset=custom,
                    bt_timestamps=None,
                    topk=5,
                    bt_nr_bins=10,
                    verbose=True,
                    labels=numeric_labels_2020)
    res = trainer.train()
    results.append(res)

#import pickle

with open(custom_path / "results.txt", "w") as file:
    file.write("\n".join(str(item) for item in results))
file.close()










# 8640 * 50 / 3600 120 hours => need smaller gridsearch
# results = pd.DataFrame(columns = ["neighbors", "components", "topics", "outliers"])
# for n in nbors:
#     for c in ncomp:
#         umap_model =  UMAP(n_neighbors=n, n_components=c, metric='cosine', low_memory=False, random_state=42)
#         cluster_model = HDBSCAN(min_cluster_size=100, metric='euclidean', prediction_data=True)
#         ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
#         # Fit BERTopic without actually performing any clustering
#         topic_model= BERTopic(
#                 embedding_model=embedding_model,
#                 umap_model=umap_model,
#                 hdbscan_model=cluster_model,
#                 ctfidf_model=ctfidf_model,
#                 verbose=False,
#         )
#         topics, _ = topic_model.fit_transform(docs, embeddings_2020)
#         t = len(set(topics))
#         topic_info = topic_model.get_topic_info()
#         # num outliers
#         o = len([x for x in topics if x == -1])
#         results.loc[len(results), :] = [n, c, t, o]
#         print(f"Neighbors: {n}, Components: {c}, Topics: {t}, Outliers: {o}")


# https://github.com/MaartenGr/BERTopic/issues/1423
# TODO: obtain proper preprocessing script
# TODO: clean up codebase
# TODO: investigate outliers
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

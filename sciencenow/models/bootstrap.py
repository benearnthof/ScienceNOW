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

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# precompute embeddings
# embeddings_2020 = embedding_model.encode(docs_2020, show_progress_bar=True)

# neighbors: 5, 10 , 15, 20
# n_components: 2:20
# min_cluster_size: 10, 20, 30, ..., 250
# TODO: evaluate scores => obtain corpus and vocab needed for octis etc. 
nbors = range(10, 20, 2)
# for anything more than 10 components the clustering saturates at around 65 clusters
# 5 neighbors and min cluster size of 100
ncomp = range(5, 11, 2) 
mclust = range(25, 300, 25)

len(nbors) * len(ncomp) * len(mclust) * 50 / 3600
from bertopic import BERTopic

params = {
    "verbose": True,
    "umap_model": UMAP(n_neighbors = 15, n_components=5, metric='cosine', low_memory=False, random_state=42),
    "hdbscan_model": HDBSCAN(min_cluster_size=1000, metric='euclidean', prediction_data=True),
    "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True)
}


tm = BERTopic(**params)

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
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
#_, timestamps = data_loader.load_docs()
#

embeddings = sentence_model.encode(data, show_progress_bar=True)

topics, _ = tm.fit_transform(data, embeddings)

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
from tqdm import tqdm
results = []
len(range(10,21,1)) * len(range(25, 325, 25)) * 60 / 3600
for neigh in tqdm(range(10,21,1)):
    for clust in range(25, 325, 25):
        params = {
        "verbose": False,
        "umap_model": UMAP(n_neighbors = neigh, n_components=5, metric='cosine', low_memory=False, random_state=42),
        "hdbscan_model": HDBSCAN(min_cluster_size=clust, metric='euclidean', prediction_data=True),
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


https://github.com/MaartenGr/BERTopic/issues/1423
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

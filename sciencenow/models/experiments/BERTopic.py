"""
DEPRECATED
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
import pickle

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
) = setup()


class ArxivProcessor:
    def __init__(self, path=ARXIV_PATH, sorted=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path
        self.sorted = sorted

    def _get_data(self):
        with open(self.path, "r") as file:
            for line in file:
                yield line

    def _process_data(self):
        date_format = "%a, %d %b %Y %H:%M:%S %Z"
        data_generator = self._get_data()
        ids, titles, abstracts, cats, refs, timestamps = [], [], [], [], [], []
        for paper in tqdm(data_generator):
            paper_dict = json.loads(paper)
            ids.append(paper_dict["id"])
            titles.append(paper_dict["title"])
            abstracts.append(paper_dict["abstract"])
            cats.append(paper_dict["categories"])
            refs.append(paper_dict["journal-ref"])
            timestamps.append(paper_dict["versions"][0]["created"])  # 0 should be v1
        # process timestamps so that they can be sorted
        timestamps_datetime = [
            datetime.strptime(stamp, date_format) for stamp in timestamps
        ]
        out = pd.DataFrame(
            {
                "id": ids,
                "title": titles,
                "abstract": abstracts,
                "categories": cats,
                "references": refs,
                "timestamp": timestamps_datetime,
            }
        )
        if self.sorted:
            return out.sort_values("timestamp", ascending=False)
        return out


arxiv = ArxivProcessor()
df = arxiv._process_data()


# print(len(ids) / index)
# about 30% of the data

# for ref in refs[0:100]:
#    print(f"{ref}")

# about 60& of papers dont have a journal reference
# using timestamps of versions as a proxy for release year

# inspect available categories
cat_list = df["categories"].unique()
print(len(cat_list))  # 76750 unique categories for 2 million articles
cat_list[0:10]

# there should be some hierarchy to these clusters
# restricting ourselves to computer science related articles
cs_df = df[df["categories"].str.contains("cs.")]
print(len(cs_df))  # 13738 Articles are left over
print(len(cs_df["categories"].unique()))  # 2842 Categories are now left

sentence_list = cs_df["abstract"].tolist()[0:5000]
years_list = cs_df["timestamp"].tolist()[0:5000]
print(sentence_list[0])

sentence_list_full = df["abstract"].tolist()
timestamp_list_full = df["timestamp"].tolist()

with open("sentence_list_full.pkl", "wb") as file:
    pickle.dump(sentence_list_full, file)

with open("timestamp_list_full.pkl", "wb") as file:
    pickle.dump(timestamp_list_full, file)

with open("sentence_list_full.pkl", "rb") as file:
    sentence_list_full = pickle.load(file)

with open("timestamp_list_full.pkl", "rb") as file:
    timestamp_list_full = pickle.load(file)

print(len(sentence_list_full), len(timestamp_list_full))

# Questions we need to answer:
#   Impact of the Embedding Model: BERT vs SentenceBERT vs Document level Model.
#   Hierarchical Topic Modeling
#   Dynamic Topic Modeling
#   Online Topic Models
#   Supervised/Semisupervised Topic Modeling
# Wrap everything up with neat visualizations
# Export trained models for deployment


# The Topic Modeling starts here
sample = sentence_list_full[0:10000]
sample.to_pickle("sample.pkl")
sample = pd.read_pickle("sample.pkl")

from sklearn.feature_extraction.text import CountVectorizer

# adjustments for large amount of documents:
# min_df => minimum frequency of words. Setting it to > 1 reduces memory overhead
# low_memory => low memory UMAP
# calculate_probabilities: False => prevents large document-topic probability matrix from being created
#
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=20)
topic_model = BERTopic(calculate_probabilities=False, low_memory=True)

topics, _ = topic_model.fit_transform(sample)

# Time for a sample of 1000 entries: 1 minute 30 seconds
# Time for a sample of 10000 documents with no adjustments for sparsity: 12 Minutes
# Time for a sample of 10000 documents with adjustments for sparsity (CPU, min_df = 10): 13 Minutes
# Time for a sample of 10000 documents with adjustments for sparsity (CPU, min_df = 20): 13 Minutes
#### Tackling very large datasets idea:
# Precalculate Embeddings
# Precalculate Vocab
# USE GPU Acceleration for HDBSCAN and UMAP
# Save model with Safetensor serialization
## Additional Tricks:
# Pre-reduce dimensionality of the embeddings
# Perform Manual topic modeling after we pre-compute clusters with HDBscan
# Manual BERTopic: if we already have pre-computed labels we dont have to calculate clusters =>
#   skips over embedding, dim-reduction, and clustering steps
# Fused embeddings: Give model the full embeddings and create customdimensionality reduction class
#   that will return reduced embeddings. This gives us the best of both worlds.
# In general: GPU required!
# => Unicluster should be powerful enough

#### Advanced techniques for large datasets:
# precalculating embeddings
import locale

locale.getpreferredencoding = lambda: "UTF-8"
from sentence_transformers import SentenceTransformer
from tempfile import gettempdir

sample = sentence_list_full
docs = [doc for doc in sample]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    docs, show_progress_bar=True
)  # 13 Minutes 1 million documents => 22 hours on CPU
# 40 minutes on A100 GPU for 2mil docs

# save embeddings to disk
with open("embeddings2M.npy", "wb") as file:
    np.save(file, embeddings)
# about 3.1GB for 2M docs, not too bad

embeddings = np.load("embeddings2M.npy")

# Preparing vocabulary
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# Extract vocab to be used in BERTopic
vocab = collections.Counter()
tokenizer = CountVectorizer().build_tokenizer()
for doc in tqdm(docs):
    vocab.update(tokenizer(doc))

with open("vocab2m.pkl", "wb") as file:
    pickle.dump(vocab, file)

with open("vocab2m.pkl", "rb") as file:
    vocab = pickle.load(file)

# 2.5 minutes give or take
len(vocab)
# 844361 words for full dataset
vocab_reduced = [word for word, frequency in vocab.items() if frequency >= 50]
len(vocab_reduced)
# >= 15: 118485 words
# >= 25: 89457 words
# >= 50: 62436 words

# training bertopic with cuml
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from bertopic import BERTopic

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP(
    n_components=5, n_neighbors=50, random_state=42, metric="cosine", verbose=True
)
hdbscan_model = HDBSCAN(
    min_samples=20,
    gen_min_span_tree=True,
    prediction_data=False,
    min_cluster_size=20,
    verbose=True,
)
vectorizer_model = CountVectorizer(vocabulary=vocab_reduced, stop_words="english")

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    verbose=True,
).fit(docs, embeddings=embeddings)

# 24 Minutes to Reduce dimensionality
# => Next step: Precompute UMAP embeddings

umap_model = UMAP(
    n_components=5, n_neighbors=15, random_state=42, metric="cosine", verbose=True
)
reduced_embeddings = umap_model.fit_transform(embeddings)

umap_model = UMAP(
    n_components=2, n_neighbors=15, random_state=42, metric="cosine", verbose=True
)
reduced_embeddings_2d = umap_model.fit_transform(embeddings)

with open("reduced_embeddings2M.npy", "wb") as file:
    np.save(file, reduced_embeddings)

with open("reduced_embeddings2M_2d.npy", "wb") as file:
    np.save(file, reduced_embeddings_2d)

# Precomputing clusters with HDBSCAN and then performing manual topic modeling

from cuml.cluster import HDBSCAN

# Find clusters of semantically similar documents
hdbscan_model = HDBSCAN(
    min_samples=30,
    gen_min_span_tree=True,
    prediction_data=False,
    min_cluster_size=30,
    verbose=True,
)
clusters = hdbscan_model.fit(reduced_embeddings).labels_

with open("hdb_clusters2M.npy", "wb") as file:
    np.save(file, clusters)

# Performing Manual BERTopic

from cuml.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired


class Dimensionality:
    """Use this for pre-calculated reduced embeddings"""

    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings

    def fit(self, X):
        return self

    def transform(self, X):
        return self.reduced_embeddings


# Prepare sub-models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = Dimensionality(reduced_embeddings)
hdbscan_model = BaseCluster()
vectorizer_model = CountVectorizer(vocabulary=vocab_reduced, stop_words="english")
representation_model = KeyBERTInspired()

# Fit BERTopic without actually performing any clustering
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    verbose=True,
).fit(docs, embeddings=embeddings, y=clusters)

# save model to disk
topic_model.save(
    path="./model_dir",
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)

topic_model = BERTopic.load("./model_dir")

# Visualizing documents
import itertools
import pandas as pd

# Define colors for the visualization to iterate over
colors = itertools.cycle(
    [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
        "#ffffff",
        "#000000",
    ]
)
color_key = {
    str(topic): next(colors) for topic in set(topic_model.topics_) if topic != -1
}

# Prepare dataframe and ignore outliers
df = pd.DataFrame(
    {
        "x": reduced_embeddings_2d[:, 0],
        "y": reduced_embeddings_2d[:, 1],
        "Topic": [str(t) for t in topic_model.topics_],
    }
)
df["Length"] = [len(doc) for doc in docs]
df = df.loc[df.Topic != "-1"]
df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
df["Topic"] = df["Topic"].astype("category")

# Get centroids of clusters
mean_df = df.groupby("Topic").mean().reset_index()
mean_df.Topic = mean_df.Topic.astype(int)
mean_df = mean_df.sort_values("Topic")

import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(16, 16))
sns.scatterplot(
    data=df,
    x="x",
    y="y",
    c=df["Topic"].map(color_key),
    alpha=0.4,
    sizes=(0.4, 10),
    size="Length",
)

# Annotate top 50 topics
texts, xs, ys = [], [], []
for row in mean_df.iterrows():
    topic = row[1]["Topic"]
    name = " - ".join(list(zip(*topic_model.get_topic(int(topic))))[0][:3])
    if int(topic) <= 50:
        xs.append(row[1]["x"])
        ys.append(row[1]["y"])
        texts.append(
            plt.text(
                row[1]["x"],
                row[1]["y"],
                name,
                size=10,
                ha="center",
                color=color_key[str(int(topic))],
                path_effects=[pe.withStroke(linewidth=0.5, foreground="black")],
            )
        )

# Adjust annotations such that they do not overlap
adjust_text(
    texts,
    x=xs,
    y=ys,
    time_lim=1,
    force_text=(0.01, 0.02),
    force_static=(0.01, 0.02),
    force_pull=(0.5, 0.5),
)
plt.show()
plt.savefig("scatterplot_2M.png", dpi=600)


#### Online Topic Modeling
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer, ClassTfidfTransformer
from river import stream
from river import cluster

docs100k = docs[0:100_000]
timestamps100k = timestamps[0:100_000]

# Prepare sub-models that support online learning
umap_model = IncrementalPCA(n_components=5)
cluster_model = MiniBatchKMeans(n_clusters=50, random_state=0)
vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=0.01)

doc_chunks = [docs100k[i : i + 1000] for i in range(0, len(docs100k), 1000)]

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
)

for docs in doc_chunks:
    topic_model.partial_fit(docs)

len(topic_model.topics_)

topic_model.visualize_topics(topic_model.topics_[0])

# continuously updating model


class River:
    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)
        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)
        self.labels_ = labels
        return self


# Using DBSTREAM to detect new topics as they come in
cluster_model = River(cluster.DBSTREAM())
vectorizer_model = OnlineCountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

# Prepare model
topic_model = BERTopic(
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
)


# Incrementally fit the topic model by training on 1000 documents at a time
for docs in doc_chunks:
    topic_model.partial_fit(docs)

# this is really slow since we do need to manually compute embeddings at the moment






























#### Dynamic Topic Modeling: How do topics change over time?


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
topic_model = BERTopic(verbose=True)
sentence_list[0]
years_list[0]

topics, probs = topic_model.fit_transform(sentence_list)

topics_over_time = topic_model.topics_over_time(sentence_list, years_list, nr_bins=7)

plot = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)

plot.show()


#### Online Topic Modeling
doc_chunks = [sentence_list[i : i + 500] for i in range(0, len(sentence_list), 500)]
# setup incremental versions of clustering, pca & vectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

# Prepare sub-models that support online learning
umap_model = IncrementalPCA(n_components=5)
cluster_model = MiniBatchKMeans(n_clusters=50, random_state=0)
vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=0.01)

from bertopic import BERTopic

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
)

# Incrementally fit the topic model by training on 1000 documents at a time
for docs in doc_chunks:
    topic_model.partial_fit(docs)

# TODO: Adjust learning rate based on stage we are in.
# Idea: Pretrain/Initial fit on large corpus of documents
#       Then Finetune on new documents

# how can we obtain new clusters when new documents come in?
# It is not guaranteed that new documents are part of established clusters
from river import stream
from river import cluster


class River:
    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self


# Using DBSTREAM to detect new topics as they come in
cluster_model = River(cluster.DBSTREAM())
vectorizer_model = OnlineCountVectorizer(stop_words="english")
from bertopic.vectorizers import ClassTfidfTransformer

ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

# Prepare model
topic_model = BERTopic(
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
)


# Incrementally fit the topic model by training on 1000 documents at a time
for docs in doc_chunks:
    topic_model.partial_fit(docs)


# Staged Approach:  Calculate Embeddings first
#                   Update Vocabulary for new documents
#                   Run Partial fit to update model




# Question: Is it possible to detect pseudo time for outlier/trend detection 
# We have embedddings and know that using UMAP we can already extract meaningful clusters
# how can we best model the "dynamics" of the underlying system

# Data analysis methods used in Single Cell RNA-Seq already try to tackle this problem
# Embeddings from the same organism (arxiv abstracts) develop over time
# a snapshot of the state of every cell (paper) is captured during sequencing
# can we use similar methods to extract trends & dynamics comparable to the 
# single cell development dynamics? 

# load variables with by running setup()
edocs = docs[0:10000]
ets = timestamps[0:10000]
embeds = embeddings[0:10000]
clust = clusters[0:10000]

edocs = [x for x, y in zip(edocs, clust) if y != -1]
ets = [x for x, y in zip(ets, clust) if y != -1]
embeds = [x for x, y in zip(embeds, clust) if y != -1]
clust = [x for x, y in zip(clust, clust) if y != -1]

from collections import Counter
cnt = Counter(clust)
assert len(edocs) == len(ets) == len(embeds) == len(clust) == 2968

# filter out the papers that dont belong to any larger group of topics
edocs = [x for x, y in zip(edocs, clust) if cnt[y] > 1]
ets = [x for x, y in zip(ets, clust) if cnt[y] > 1]
embeds = [x for x, y in zip(embeds, clust) if cnt[y] > 1]
clust = [x for x, y in zip(clust, clust) if cnt[y] > 1]

cnt = Counter(clust)

X = np.array(embeds)
obs = np.array(clust)

import scanpy
import scvelo as scv
from anndata import AnnData

adata = AnnData(X=X, obs=obs)
adata.layers["spliced"] = X
adata.layers["unspliced"] = X
adata.obs["cell_groups"] = obs

# scanpy diffusionmap
scanpy.pp.neighbors(adata, n_neighbors=15, use_rep="X")
scanpy.tl.diffmap(adata)
scanpy.tl.draw_graph(adata)
scanpy.pl.draw_graph(adata, color="cell_groups", save="scanpy_plot_1_filtered.png")

# pseudo time for diffusionmap
# https://training.galaxyproject.org/training-material/topics/single-cell/tutorials/scrna-case_JUPYTER-trajectories/tutorial.html
# root is the "oldest" paper, we are interested in the time flow from 
# this group to the oters


adata.uns['iroot'] = np.flatnonzero(adata.obs['cell_groups']  == 1902)[0]

scanpy.tl.dpt(adata)
scanpy.pl.draw_graph(adata, color=['cell_groups', 'dpt_pseudotime'], legend_loc='on data', save = 'DPT.png')

# now lets compute pseudo velocity with scvelo
adata = AnnData(X=X, obs=obs)
adata.layers["spliced"] = X
adata.layers["unspliced"] = X
adata.obs["cell_groups"] = obs

scv.utils.show_proportions(adata)

scv.pp.normalize_per_cell(adata, enforce=True)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)


scv.tl.velocity(adata, mode="dynamical")
scv.pl.velocity(adata)

# experimenting with UMAP


# Dynamic Topic Modeling

(
    docs,
    timestamps,
    embeddings,
    vocab,
    reduced_embeddings,
    reduced_embeddings_2d,
    clusters,
) = setup(n=50000)

from bertopic import BERTopic

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs)
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=50)
# runs about 14sec for 10k articles
# global_tuning and evolutionary_tuning may have additional effects
tovert_nonglobal = topic_model.topics_over_time(docs, timestamps, global_tuning=False, evolution_tuning=True, nr_bins=20)
# we should use binning to keep the number of unique topic representations under control
topics_over_time_20bins = topic_model.topics_over_time(docs, timestamps, nr_bins=20)

plot = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
plot.write_image("topics_over_time_20bins.png")
plot = topic_model.visualize_topics_over_time(topics_over_time_20bins, top_n_topics=10)
plot.write_image("topics_over_time_10bins.png")

similar_topics, similarity = topic_model.find_topics("NLP", top_n=5)
topic_model.get_topic(similar_topics[0])

# visualize NLP over time
plot = topic_model.visualize_topics_over_time(topics_over_time, topics=similar_topics)
plot.write_image("topics_over_time_nlp.png")

# custom topic labels
topic_labels = topic_model.generate_topic_labels(nr_words=2,topic_prefix=False)
topic_model.set_topic_labels(topic_labels)

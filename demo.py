"""
Preprocessing a fresh OAI snapshot.

To preprocess a new snapshot or set everything up for the first time save a config like so:

ARXIV_SNAPSHOT: "path/to/new/snapshot/arxiv-metadata-oai-2023-11-13.json"
EMBEDDINGS: "path/for/new/embeddings.npy"
REDUCED_EMBEDDINGS: "path/for/new/reduced_embeddings.npy"
FEATHER_PATH: "path/for/new/arxiv_df.feather"
TAXONOMY_PATH: "path/to/taxonomy/for/semisupervised/model/taxonomy.txt"
EVAL_ROOT: "path/to/store/eval/results/"
VOCAB_PATH: "path/for/new/vocab.txt"
CORPUS_PATH: "path/for/new/corpus.tsv"
SENTENCE_MODEL: "sentence-transformers/all-distilroberta-v1"
UMAP_NEIGHBORS: 15
UMAP_COMPONENTS: 5
UMAP_METRIC: "cosine"
VOCAB_THRESHOLD: 15
TM_VOCAB_PATH: "path/for/new/tm_vocab.txt" (used to cache vocab during evaluation)
TM_TARGET_ROOT: "path/for/new/eval" (used to cache topic models during evaluation)

"""

#### Preprocessing Data
# Loading the snapshot first
from sciencenow.data.arxivprocessor import ArxivProcessor
from sciencenow.models.train import ModelWrapper
from sciencenow.config import (
    setup_params,
    online_params
)

from sciencenow.postprocessing.trends import (
    TrendValidatorIR,
    TrendExtractor
)
import numpy as np


processor = ArxivProcessor(sort_by_date=True)
processor.load_snapshot()

# Embed with the sentence transformer of choice
# If you're running this for the first time, the processor will download the respective model config and weights to disk first
# The processor will then precalculate batches depending on the power of your GPU.
# This will also take around 1-2 minutes
# after this all 2.7 million abstracts will be encoded to embeddings and saved disk at
# "path/for/new/embeddings.npy"
# Depending on which model you've chosen the entire process will take anywhere from 40-160 minutes.
# Since we already precomputed the embeddings this method will simply load the embeddings from disk.
processor.embed_abstracts()
# The processor will automatically save the new raw embeddings to the location specified in the config, make sure you have at least 8GB disk space available

# Next we will run the dimensionality reduction
# Without subset since we want to reduce the entire corpus
# And without labels since for semisupervised models we will have to recompute the UMAP step anyway.
# This will take about 10 minutes for 2.3 million documents
# We will obtain labels first so we can perform semisupervised models without having to recalculate the embeddings every time
# (only for deployment, for evaluation we should recalculate since umap depends on neighborhood structure)
# first we need plaintext labels we can convert to numeric labels for umap
subset = processor.filter_by_taxonomy(subset=processor.arxiv_df, target=None, threshold=0)
plaintext_labels, numeric_labels = processor.get_numeric_labels(subset, 0)

# This is usually done in BERTopic setup but we do it manually since we only call reduce_embeddings here
ids = subset.index.tolist()
processor.subset_embeddings = processor.embeddings[ids]

processor.reduce_embeddings(subset=subset, labels=numeric_labels) # This takes about 10 minutes on an A100 40GB
np.save(processor.FP.REDUCED_EMBEDS.value, processor.subset_reduced_embeddings, allow_pickle=False)
# These are all the preprocessing steps necessary for running topic models on huge datasets
# But this can also be performed by just using the ModelWrapper class since it instantiates a processor anyway. 


#### Using the Model Wrapper for Setup:
# what happens if we just let the model wrapper do its thing?

setup_params["target"] = "cs.LG"
setup_params["cluster_size"] = 6
setup_params["secondary_target"] = "q-bio"
setup_params["secondary_proportion"] = 0.2
setup_params["recompute"] = True
wrapper = ModelWrapper(setup_params=setup_params, model_type="semisupervised", usecache=False)

# This does the following: 
# We look for a preprocessed .feather file that is a lot quicker to load from disk than a raw dataframe
# We look for a cached subset, in case we already ran a model for the same date range
# we load the sentence transformer embeddings of all abstracts
# we select the relevant embeddings for the docuemnts that fall in the date range we specified
# we recompute the reduced embeddings for this subset (if recompute: true)
#   For Semisupervised Models we have no choice but to recompute embeddings, since we need to consider the labels in the UMAP step.
# we build a corpus.tsv and vocab.txt file for the topic model evaluation

# we can now train and evaluate a topic model with the wrapper we just instantiated: 
# for evaluation it is recommended to set setup_params["limit"] to a value between 5000 and 7500
wrapper.tm_setup()
_ = wrapper.tm_train()
# inspecting Diversity and Coherence: 

# with wrapper.topics we can check how many papers of the additional group got assigned to a topic

ds = wrapper.subset.copy()
ds["topic"] = wrapper.topics

ds_synthetic = ds[ds["l1_labels"] == "q-bio.PE"]
counts_synthetic = ds_synthetic.groupby("topic").size()
ds_background = ds[ds["l1_labels"] == "cs.LG"]
counts_background = ds_background.groupby("topic").size()



# let's inspect background papers

# without recompute 31 qbio papers are sorted into background class -1
# 31 are sorted into class 0 with data learning models deep as info
# the rest are sparsely distributed across 17 other classes

# with recompute only 15 of the 102 synthetic papers get sorted into background class
# only two papers get put into groups they are technically not a member of
# the rest are all in pure, separate clusters

# inspecting papers that fell into background class
bg = ds_synthetic[ds_synthetic["topic"] == -1].title.tolist()
# they are all related to statistical modelling of pandemics and outbreaks, so pretty realistic for them to be grouped up in a background of 
# ['information', 'domains', 'series', 'patient', 'sample', 'properties', 'samples', 'features', 'proposed', 'architecture']



# Run the visualization with the original embeddings
# out = wrapper.topic_model.visualize_documents(
#     wrapper.subset.title.tolist(),
#     embeddings=wrapper.processor.subset_reduced_embeddings
# )

# out.write_html("C:\\Users\\Bene\\Desktop\\norecompute_q-bio_vs_cs.LG.html")



# let's see if we can identify some trends

# wrapper.train()
extractor = TrendExtractor(model_wrapper=wrapper)

deviations = extractor.calculate_deviations()

candidates = extractor.get_candidate_papers(
    subset=wrapper.subset,
    topics=wrapper.topics,
    deviations=deviations,
    threshold=1.5)


# we can extract trends with the trend extractor but need to compare them with the evaluator
# we know a priori how the synthetic trend was injected, we validate with the validator 
# we have the target timestamps sorted
target_timestamps = ds_synthetic.v1_datetime.tolist()
# and can obtain the papers per bin that were sampled from the wrapper
papers_per_bin = wrapper.papers_per_bin
# papers are already sorted by timestamp
# so we can just select the correct number of papers from the top of the df as our targets for the respective time stamps

# now we extract the trends with the basic trend extractor
# and evaluate overlap of all trending classes with the target set

import numpy as np

def extract_max_papers(dataframe, papers_per_bin):
    cumulative_sums = np.cumsum(papers_per_bin)
    max_index = np.argmax(cumulative_sums)
    if max_index == 0:
        start_index = 0
    else:
        start_index = cumulative_sums[max_index - 1]
    end_index = cumulative_sums[max_index]
    return dataframe.iloc[start_index:end_index]

target_set = extract_max_papers(ds_synthetic, papers_per_bin)

target_ids = set(target_set.id.tolist())

def extract_ids(dictionary):
    id_dict = {}
    for key, list_of_dicts in dictionary.items():
        for item in list_of_dicts:
            if key not in id_dict:
                id_dict[key] = []
            id_dict[key].append(item['id'])
    return {key: set(value) for key, value in id_dict.items()}

candidate_ids = extract_ids(candidates)

# now get background ids to distinguish basic trends from candidates that were missed
background_ids = set(ds_background.id.tolist()).difference(target_ids)
synth_background_ids = set(ds_synthetic.id.tolist()).difference(target_ids)

validator = TrendValidatorIR(
    results=candidate_ids,
    background_docs=background_ids,
    synth_background_docs=synth_background_ids,
    target_docs=target_ids
)

precisions = [validator.precision_at_k(k) for k in range(len(candidate_ids))]
dcg_at_ks = [validator.dcg_at_k(k) for k in range(len(candidate_ids))]

# overall performance: How many target ids could be found in trending clusters
trending_topics = [x for x in candidate_ids]
counts_synthetic
counts_background

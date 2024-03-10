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
RECOMPUTE_ALL = False
#### Preprocessing Data
# Loading the snapshot first
from sciencenow.models.train import ModelWrapper
from sciencenow.config import (
    setup_params,
    online_params
)

from sciencenow.postprocessing.trends import (
    TrendValidatorIR,
    TrendExtractor,
    TrendPostprocessor
)

from sciencenow.utils.plotting import plot_trending_papers
import numpy as np

if RECOMPUTE_ALL:
    from sciencenow.data.arxivprocessor import ArxivProcessor
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
setup_params["trend_deviation"] = 1.67
setup_params["recompute"] = True
# if we somehow adjusted setup_params we can set usecache to False to guarantee refiltering (not necessary 99% of the time)
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

#### Trend Extraction: 

trend_postprocessor = TrendPostprocessor(wrapper=wrapper)
# Performance calculations can be done since we added a synthetic subset
trend_postprocessor.calculate_performance(threshold=1.5)

trend_df, trend_embeddings = trend_postprocessor.get_trend_info(threshold=1.5)

# Inspect trend_df to see which papers were identified as trending
# We can now visualize the results
fig = plot_trending_papers(trend_df, trend_embeddings)
fig.write_html("C:\\Users\\Bene\\Desktop\\trends2.html")

# TODO: compare to online model

"""
Verify that preprocessing works as intended.
"""
from collections import Counter
from sciencenow.data.arxivprocessor import ArxivProcessor

#### TEST SETUP
# IF
processor = ArxivProcessor()
# WHEN
processor.load_snapshot()
# THEN
assert processor.arxiv_df.shape == (2363326, 14)
assert processor.embeddings == None
assert processor.reduced_embeddings == None
assert processor.taxonomy == None
assert all(fp.value.exists() for fp in processor.FP)
assert processor.PARAMS.SENTENCE_MODEL.value == "all-MiniLM-L6-v2"
assert processor.PARAMS.UMAP_COMPONENTS.value == 5
assert processor.PARAMS.UMAP_NEIGHBORS.value == 15
assert processor.PARAMS.UMAP_METRIC.value == "cosine"

#### TEST FILTERING
# IF 
startdate = "01 01 2020"
enddate = "31 12 2020"
# WHEN
subset = processor.filter_by_date_range(startdate=startdate, enddate=enddate)
# THEN
assert subset.shape == (178275, 14)
assert subset.abstract[0:1].tolist()[0].startswith("Let $X$ be a compact Calabi-Yau 3-fold")
assert subset.id[0:1].tolist()[0] == "2001.00113"


#### TEST FILTERING BY LABEL
# TEST 0 THRESHOLD
# IF 
target = "cs"
threshold = 0
subset = processor.filter_by_date_range(startdate=startdate, enddate=enddate)
# WHEN 
subset = processor.filter_by_taxonomy(subset=subset, target=target, threshold=threshold)
# THEN
assert subset.shape == (63530, 17)
assert len(Counter(subset.plaintext_labels)) == 2939
assert all([val > threshold for val in Counter(subset.plaintext_labels).values()])

# TEST DIFFERENT TARGET
# IF 
target = "math"
threshold = 100
# WHEN 
subset = processor.filter_by_date_range(startdate=startdate, enddate=enddate)
subset = processor.filter_by_taxonomy(subset=subset, target=target, threshold=threshold)
# THEN
assert subset.shape == (27061, 17)
assert len(Counter(subset.plaintext_labels)) == 43
assert all([val > threshold for val in Counter(subset.plaintext_labels).values()])
labs = [label.split(" ") for label in subset.l1_labels.tolist()]
# verify that all documents with only one label are labeled as "math"
all([x.startswith(target) for y in labs for x in y if len(y) == 1])
# verify that all documents with more than one label have at least one "math" label
any([x.startswith(target) for y in labs for x in y if len(y) > 1])

# IF 
target = "cs"
threshold = 100
# WHEN 
subset = processor.filter_by_date_range(startdate=startdate, enddate=enddate)
subset = processor.filter_by_taxonomy(subset=subset, target=target, threshold=threshold)
# THEN
assert subset.shape == (49380, 17)
assert len(Counter(subset.plaintext_labels)) == 77
assert all([val > threshold for val in Counter(subset.plaintext_labels).values()])

#### TEST EMBEDDING LOADING
# IF
# WHEN
embeddings = processor.load_embeddings()
ids = subset.index.tolist()
subset_abstracts = subset.loc[ids].abstract
original_abstracts = processor.arxiv_df.loc[ids].abstract
processor.bertopic_setup(subset=subset)

# THEN
assert processor.embeddings.shape[0] == processor.arxiv_df.shape[0]
assert all([x == y for x, y in zip(subset_abstracts,original_abstracts)])
assert processor.subset_reduced_embeddings.shape == (subset.shape[0], 5)

#### TEST EMBEDDING RECOMPUTE
# IF 
processor.bertopic_setup(subset=subset, recompute=True)
# WHEN 
recomp_embeds = processor.subset_reduced_embeddings
processor.bertopic_setup(subset=subset, recompute=False)
preload_embeds = processor.subset_reduced_embeddings
# THEN 
# unpack all array entries and compare every single entry to its preloaded counterpart
assert all([w for z in [x != y for x, y in zip(recomp_embeds, preload_embeds)] for w in z])

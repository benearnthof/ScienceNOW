from pathlib import Path
from tempfile import TemporaryFile
from numpy import ndarray

from sciencenow.core.pipelines import (
    PubmedPipeline,
    ArxivPipeline,
)
from sciencenow.core.steps import (
    PubmedLoadStep,
    PubmedPreprocessingStep,
    ArxivLoadJsonStep,
    ArxivDateTimeParsingStep,
    ArxivDateTimeFilterStep,
    ArxivAbstractPreprocessingStep,
    ArxivTaxonomyFilterStep,
    ArxivPlaintextLabelStep,
    ArxivReduceSubsetStep,
    ArxivGetNumericLabelsStep,
    ArxivSaveFeatherStep,
    ArxivLoadFeatherStep,
)

from sciencenow.core.dataset import (
    ArxivDataset,
    Dataset,
    BasicMerger,
    SyntheticMerger
)


from sciencenow.core.embedding import ArxivEmbedder
from sciencenow.core.dimensionality import UmapReducer

#### Test Pubmed Pipeline
path = Path("C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\arxiv_df.feather")
input = Path("D:\\textds\\train\\train.txt")


pipe = PubmedPipeline(
    steps=[
        PubmedLoadStep, 
        PubmedPreprocessingStep
    ]
)

out = pipe.execute(input=input)



#### Test Arxiv Pipeline
path = "C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\taxonomy.txt"
ds = ArxivDataset(path=path, pipeline=None)
ds.load_taxonomy(path=path)

input = Path("C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\arxiv-metadata-oai-2023-11-13.json")

pipe = ArxivPipeline(
    steps=[
        ArxivLoadJsonStep(nrows=None),
        ArxivDateTimeParsingStep(),
        ArxivDateTimeFilterStep(
            interval={
                "startdate": "31 03 2007",
                "enddate": "15 04 2007"}),
        ArxivAbstractPreprocessingStep(),
        ArxivTaxonomyFilterStep(target="cs"),
        ArxivPlaintextLabelStep(taxonomy=ds.taxonomy, threshold=0, target="cs"),
        ArxivReduceSubsetStep(limit=100),
        ArxivGetNumericLabelsStep(mask_probability=0),
    ]
)

output = pipe.execute(input=input)

assert len(output) == 77
assert "v1_datetime" in output.keys()
assert "l1_labels" in output.keys()
assert "plaintext_labels" in output.keys()
assert "numeric_labels" in output.keys()

# Perform preprocessing and save to .feather
pipe = ArxivPipeline(
    steps=[
        ArxivLoadJsonStep(nrows=None),
        ArxivDateTimeParsingStep(),
        ArxivAbstractPreprocessingStep(),
        ArxivSaveFeatherStep(path="C:\\Users\\Bene\\Desktop\\arxiv_processed.feather")
    ]
)

output = pipe.execute(input=input)

# This highlights the composability of the pipeline/steps pattern, as we now only have to 
# execute the feather load step and then perform filtering like so: 

pipe = ArxivPipeline(
    steps=[
        ArxivLoadFeatherStep(),
        ArxivDateTimeFilterStep(
            interval={
                "startdate": "01 01 2020",
                "enddate": "31 12 2020"}),
        ArxivTaxonomyFilterStep(target="cs"),
        ArxivPlaintextLabelStep(taxonomy=ds.taxonomy, threshold=0, target="cs"),
        ArxivReduceSubsetStep(limit=50),
        ArxivGetNumericLabelsStep(mask_probability=0),
    ]
)

output = pipe.execute(input="C:\\Users\\Bene\\Desktop\\arxiv_processed.feather")
# Now loading only causes a temporary spike in memory consumption when the dataframe is loaded.
# After filtering only the 50 relevant papers remain. 
# Let's perform loading without filtering to see the upper limit of memory consumption

path = "C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\taxonomy.txt"
ds = ArxivDataset(path=path, pipeline=None)
ds.load_taxonomy(path=path)

pipe = ArxivPipeline(
    steps=[
        ArxivLoadFeatherStep(),
        ArxivPlaintextLabelStep(taxonomy=ds.taxonomy, threshold=0, target=None),
        ArxivGetNumericLabelsStep(mask_probability=0),
    ]
)

output = pipe.execute(input="C:\\Users\\Bene\\Desktop\\arxiv_processed.feather")
# Without any filtering python sits at 4.8 GB of Memory consumption.
# For performance reasons we do want to keep the data in memory and only load the embeddings we need.
# old_output = pipe.execute(input="C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\arxiv_df.feather")
# the index entries of both data frames match => we can just pick embeddings by indexing





embedder = ArxivEmbedder(
    source=None,
    target=None,
    data=output["abstract"].tolist()[0:30],
)

embedder.embed()

assert isinstance(embedder.embeddings, ndarray)
assert embedder.embeddings.shape == (30, 768)

reducer = UmapReducer()

data = embedder.embeddings

#### Test Datasets and Dataset merger
# set up sample of 50 papers from 2020
path = "C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\taxonomy.txt"
ds = ArxivDataset(path=path, pipeline=None)
ds.load_taxonomy(path=path)

source_pipe = ArxivPipeline(
    steps=[
        ArxivLoadFeatherStep(),
        ArxivDateTimeFilterStep(
            interval={
                "startdate": "01 01 2020",
                "enddate": "31 12 2020"}),
        ArxivTaxonomyFilterStep(target="cs"),
        ArxivPlaintextLabelStep(taxonomy=ds.taxonomy, threshold=0, target="cs"),
        ArxivReduceSubsetStep(limit=50),
        ArxivGetNumericLabelsStep(mask_probability=0),
    ]
)

sourceds = ArxivDataset(path=TemporaryFile().file.name, pipeline=source_pipe)
sourceds.execute_pipeline(input="C:\\Users\\Bene\\Desktop\\arxiv_processed.feather")

# set up sample of 50 papers from 2021
target_pipe = ArxivPipeline(
    steps=[
        ArxivLoadFeatherStep(),
        ArxivDateTimeFilterStep(
            interval={
                "startdate": "01 01 2021",
                "enddate": "31 12 2021"}),
        ArxivTaxonomyFilterStep(target="cs"),
        ArxivPlaintextLabelStep(taxonomy=ds.taxonomy, threshold=0, target="cs"),
        ArxivReduceSubsetStep(limit=50),
        ArxivGetNumericLabelsStep(mask_probability=0),
    ]
)
targetds = ArxivDataset(path=TemporaryFile().file.name, pipeline=target_pipe)
targetds.execute_pipeline(input="C:\\Users\\Bene\\Desktop\\arxiv_processed.feather")

# both datasets are still ordered by v1_datetime
# setting up embedder (in this case the same one for both datasets)
embedder = ArxivEmbedder(
    source="C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\embeddings.npy",
    target=TemporaryFile().file.name,
    data=None
)

embedder.load()
# Setting up the merger class
merger = BasicMerger(
    source=sourceds,
    target=targetds,
    source_embedder=embedder,
    target_embedder=embedder,
)

merger.merge()

assert len(merger.data) == 100
assert merger.embeddings.shape == (100, 768)

# We pass the merger object to the Model Object with the Reducer object.
# The Model will take care of adjusting the labels and then performing the dim reduction.

merger = SyntheticMerger(
    source=sourceds,
    target=targetds,
    source_embedder=embedder,
    target_embedder=embedder,
)
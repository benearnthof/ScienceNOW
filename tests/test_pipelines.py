from pathlib import Path
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

from sciencenow.core.dataset import ArxivDataset

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
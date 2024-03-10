from pathlib import Path

from sciencenow.core.pipelines import (
    PubmedPipeline,
    ArxivPipeline,
)
from sciencenow.core.steps import (
    PubmedLoadStep,
    PubmedPreprocessingStep,
    ArxivLoadStep,
    ArxivDateTimeParsingStep,
    ArxivDateTimeFilterStep,
    ArxivAbstractPreprocessingStep,
    ArxivTaxonomyFilterStep,
    ArxivPlaintextLabelStep,
    ArxivReduceSubsetStep,
    ArxivGetNumericLabelsStep,
)

from sciencenow.core.dataset import ArxivDataset


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
ds = ArxivDataset(path="path", pipeline=None)
ds.load_taxonomy(path=path)

input = Path("C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\arxiv-metadata-oai-2023-11-13.json")

pipe = ArxivPipeline(
    steps=[
        ArxivLoadStep(nrows=10000),
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
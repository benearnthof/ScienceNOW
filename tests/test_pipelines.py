from pathlib import Path
from tempfile import TemporaryFile
from numpy import ndarray, array

from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer

from sciencenow.config import (
    TM_PARAMS,
    setup_params,
    online_params,
)

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
    GetOctisDatasetStep,
    GetDynamicTopicsStep,
    GetMetricsStep,
    CalculateDynamicDiversityStep,
    CalculateCoherenceStep,
    ExtractEvaluationResultsStep,
)

from sciencenow.core.dataset import (
    ArxivDataset,
    Dataset,
    BasicMerger,
    SyntheticMerger
)

from sciencenow.core.embedding import ArxivEmbedder
from sciencenow.core.dimensionality import UmapReducer
from sciencenow.core.model import BERTopicDynamic
from sciencenow.core.experiment import BERTopicExperiment

SETUP_PARAMS = setup_params

#### Testing the entire Experiment pipeline
# set up sample of 50 papers from 2020
path = "C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\taxonomy.txt"
ds = ArxivDataset(path=path, pipeline=None)
ds.load_taxonomy(path=path)

target_pipe = ArxivPipeline(
    steps=[
        ArxivLoadFeatherStep(),
        ArxivDateTimeFilterStep(
            interval={
                "startdate": "01 01 2020",
                "enddate": "31 01 2020"}),
        ArxivTaxonomyFilterStep(target="cs"),
        ArxivPlaintextLabelStep(taxonomy=ds.taxonomy, threshold=25, target="cs"),
        ArxivReduceSubsetStep(limit=400),
        ArxivGetNumericLabelsStep(mask_probability=0),
    ]
)

targetds = ArxivDataset(path=TemporaryFile().file.name, pipeline=target_pipe)
targetds.execute_pipeline(input="C:\\Users\\Bene\\Desktop\\arxiv_processed.feather")

# set up sample of q-bio papers we will add to target as a synthetic trend
source_pipe = ArxivPipeline(
    steps=[
        ArxivLoadFeatherStep(),
        ArxivDateTimeFilterStep(
            interval={
                "startdate": "01 01 2021",
                "enddate": "31 12 2021"}),
        ArxivTaxonomyFilterStep(target="q-bio"),
        ArxivPlaintextLabelStep(taxonomy=ds.taxonomy, threshold=25, target=None),
        #ArxivReduceSubsetStep(limit=50),
        ArxivGetNumericLabelsStep(mask_probability=0),
    ]
)
sourceds = ArxivDataset(path=TemporaryFile().file.name, pipeline=source_pipe)
sourceds.execute_pipeline(input="C:\\Users\\Bene\\Desktop\\arxiv_processed.feather")

# both datasets are still ordered by v1_datetime
# setting up embedder (in this case the same one for both datasets)
embedder = ArxivEmbedder(
    source="C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\embeddings.npy",
    target=TemporaryFile().file.name,
    data=None
)

embedder.load()

# We pass the merger object to the Model Object with the Reducer object.
# The Model will take care of adjusting the labels and then performing the dim reduction.

merger = SyntheticMerger(
    source=sourceds,
    target=targetds,
    source_embedder=embedder,
    target_embedder=embedder,
)

merger.merge(setup_params=SETUP_PARAMS)


# Now we instantiate a reducer object that will take care of dimensionality reduction.
reducer = UmapReducer(
    setup_params=SETUP_PARAMS,
    data=merger.embeddings,
    labels=array(merger.data.numeric_labels)
    )

reducer.reduce()

# Now we set up the topic model we wish to train

cluster_model = HDBSCAN(
                min_samples=SETUP_PARAMS["samples"],
                gen_min_span_tree=True,
                prediction_data=True,
                min_cluster_size=SETUP_PARAMS["cluster_size"],
)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

model = BERTopicDynamic(
    data = merger.data,
    embeddings = reducer.reduced_embeddings,
    cluster_model=cluster_model,
    ctfidf_model=ctfidf_model,
)

# train model

model.train(setup_params=SETUP_PARAMS)

eval_pipe = ArxivPipeline(
    steps = [
        GetOctisDatasetStep(root = str(Path(TemporaryFile().file.name).parent)),
        GetDynamicTopicsStep(),
        GetMetricsStep(),
        CalculateDynamicDiversityStep(),
        CalculateCoherenceStep(),
        ExtractEvaluationResultsStep(
            id=merger.get_id(SETUP_PARAMS, primary=False),
            setup_params=SETUP_PARAMS
        ),
    ]
)

eval_pipe.execute(input=model)
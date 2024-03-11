from pathlib import Path

from sciencenow.core.steps import (
    PubmedLoadStep,
    PubmedPreprocessingStep,
    ArxivLoadStep,
    ArxivDateTimeParsingStep,
    ArxivDateTimeFilterStep,
)


path = Path("C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\arxiv_df.feather")
input = Path("D:\\textds\\train\\train.txt")

loadstep = PubmedLoadStep()
output = loadstep.execute(input=input)

assert len(output) == 2593169
assert output[0].startswith("###")
assert output[-1] == ''

processingstep = PubmedPreprocessingStep()
output = processingstep.execute(input=output)

input = Path("C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\arxiv-metadata-oai-2023-11-13.json")
loadstep = ArxivLoadStep(nrows=10000)
output = loadstep.execute(input=input)

# TODO: Assert that every processing step returns a data frame that is ordered by v1_datetime
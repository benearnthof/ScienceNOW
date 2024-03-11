import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Union, Dict
from pathlib import Path
from collections import Counter
from dateutil import parser
from tqdm import tqdm
from pandas import DataFrame, read_json, to_datetime, read_feather
from numpy import array
from random import choices

from sciencenow.core.utils import get_plaintext_name

class Step(ABC):
    """
    Abstract Base Class for all pre- & postprocessing Steps.
    """
    @abstractmethod
    def execute(self, input: Any) -> Any:
        raise NotImplementedError


class PubmedLoadStep(Step):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def execute(input: Path) -> List[str]:
        if not input.exists():
            raise FileNotFoundError(f"No file found at {input}")
        with open(input) as file:
            output = [line.rstrip().split("\t")[1] if "\t" in line else line.rstrip() for line in file]
        return output


class PubmedPreprocessingStep(Step):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def execute(input: List[str]) -> DataFrame:
        chunks = {}
        temp = []
        # In pubmed every abstract is tagged with an ID
        # every abstract is split into lines that consist of a tag and a sentence separated by a tab.
        for line in input:
            if line.startswith("###"):
                idx = line.split("###")[1] + "PUBMED"
            elif line != "":
                temp.append(line)
            elif line == "":
                chunks[idx] = " ".join(temp)
                temp = []
            else:
                print(line)
        # we need id, title, abstract, l1_labels, plaintext_labels
        data = {
            "id": chunks.keys(),
            "abstract": chunks.values(),
            "l1_labels": "PUBMED",
            "plaintext_labels": "PUBMED",
            "categories": "PUBMED"
        }
        return DataFrame(data)
        

class ArxivLoadJsonStep(Step):
    """
    Step to load OAI snapshot from disk and return a pandas dataframe.
    """
    def __init__(self, nrows:Union[int, None]) -> None:
        super().__init__()
        self.nrows=nrows
    
    def execute(self, input: Path) -> List[str]:
        if not input.exists():
            raise FileNotFoundError(f"No file found at {input}")
        if not input.suffix == ".json":
            raise NotImplementedError(f"Snapshot must be stored in .json format, found {input.suffix}")
        output = read_json(input, lines=True, nrows=self.nrows)
        output = output.drop(
            labels=[
                "submitter", 
                "authors", 
                "comments",
                "journal-ref",
                "doi",
                "report-no",
                ], # TODO: Can we drop "versions"?
            axis=1
            )
        output = output.drop_duplicates(subset="id")
        print(f"Loaded {len(output)} entries from {input}.")
        return output
    
class ArxivLoadFeatherStep(Step):
    """
    Step to directly load a .feather file containing a dataframe with 
    ArxivLoadJsonStep
    ArxivDateTimeParsingStep
    ArxivAbstractPreprocessingStep
    Already Performed. This saves about 5 minutes of computation for the sequential datetime parsing.
    """
    def __init__(self) -> None:
        super().__init__()

    def execute(self, input: str) -> DataFrame:
        path = Path(input)
        if not path.exists():
            raise FileNotFoundError(f"No file found at specified location: {path}")
        if not path.suffix == ".feather":
            raise NotImplementedError(f"Data must be stored in .feather format, found {path.suffix}")
        print(f"Loaded dataframe from {path}")
        return read_feather(path)
    
class ArxivSaveFeatherStep(Step):
    """
    Step to directly save a .feather file containing a dataframe with 
    ArxivLoadJsonStep
    ArxivDateTimeParsingStep
    ArxivAbstractPreprocessingStep
    Already Performed to disk.
    """
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path=Path(path)

    def execute(self, input: DataFrame) -> DataFrame:
        if input is None:
            raise NotImplementedError(f"Provide Dataset to save.")
        if not isinstance(input, DataFrame):
            raise NotImplementedError(f"Input must be `pandas.core.frame.DataFrame`.")
        if not self.path.suffix == ".feather":
            raise NotImplementedError(f"Data must be stored in .feather format, found {self.path.suffix}")
        print(f"Saving dataframe to {self.path}")
        input.to_feather(self.path)
        return input
        


class ArxivDateTimeParsingStep(Step):
    """
    Step that converts raw version dates to single publication timestamps for later filtering.
    Main idea: Because papers get updated over time, they have more than one time stamp. While the initial v1 timestamp may only mark the initial release and not 
        the official publication in a journal it is still much more logical to use v1 as a proxy for release than to double check every preprint.
    """
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def execute(input: DataFrame) -> DataFrame:
        if not isinstance(input, DataFrame):
            raise NotImplementedError(f"Provide data as pandas.core.frame.DataFrame. Found {type(input)} instead.")
        # the versions_dates column contains all update dates in a list for every preprint
        versions = input.versions_dates
        versions = [x[0] for x in versions]
        print(f"Parsing {len(versions)} version 1 strings to datetime format...")
        # parse to date time format for easier filtering and sorting
        versions = [parser.parse(dt) for dt in tqdm(versions)]
        # insert parsed v1_dates as new column
        input["v1_datetime"] = versions
        return input.sort_values("v1_datetime")
    

class ArxivDateTimeFilterStep(Step):
    """
    Step that returns a subset of the data based on a given start and end date time.
    Will exclusively filter data from 00:00:00 GMT of startdate to 23:59:00 GMT of enddate.

    Args:
        startdate: string of the form "day month year" with their respective numeric values
        enddate: string of the form "day month year" with their respective numeric values
            each seperated by spaces. Example: "31 12 2020" corresponds to the 31st of December 2020.
    """
    def __init__(self, interval: Dict[str,str]) -> None:
        super().__init__()
        self.interval = interval

    def execute(self, input:DataFrame) -> DataFrame:
        startdate, enddate = self.interval["startdate"], self.interval["enddate"]
        if startdate is None:
            warnings.warn("No date range selected, returning full dataframe.")
            return(input)
        if not "v1_datetime" in input.keys():
            raise NotImplementedError("Input has no datetime column. Execute `ArxivDateTimeParsingStep` first.")
        a, b, c = startdate.split(" ")
        # TODO: Why does this only work if we swap b and a for start and end?
        # this actually caused the out of memory errors and other downstream problems, the swap is needed.
        start = to_datetime(parser.parse(f"{b} {a} {c} 00:00:01 GMT"))
        a, b, c = enddate.split(" ")
        end = to_datetime(parser.parse(f"{a} {b} {c} 23:59:59 GMT"))
        # computing mask takes a moment if the dataframe has lots of rows
        mask = [(date > start) & (date < end) for date in tqdm(input["v1_datetime"])]
        return input.loc[mask]
        

class ArxivAbstractPreprocessingStep(Step):
    """
    Step that removes newline characters from Abstracts
    """
    def __init__(self) -> None:
        super().__init__()

    def execute(self, input: DataFrame) -> DataFrame:
        if not isinstance(input, DataFrame):
            raise NotImplementedError(f"Provide data as pandas.core.frame.DataFrame. Found {type(input)} instead.")
        # remove newline characters and strip leading and traling spaces.
        docs = [doc.replace("\n", " ").strip() for doc in input["abstract"].tolist()]
        input["abstract"] = docs
        print(f"Successfully removed newlines, leading, and traling spaces from {len(docs)} abstracts.")
        return input
    

class ArxivTaxonomyFilterStep(Step):
    """
    Optional step to filter a subset of papers by soft class labels from Arxiv Taxonomy.
    Will filter a dataframe witih "categories" to only contain articles tagged as the desired category.
    Because every paper has potentially many different labels this function will filter by simple 
    majority vote. If >= 50% of labels for a paper match the specified target string, the paper
    will be kept in the output subset.
    
    Args: 
        input: dataframe that should be filtered (already filtered by timeperiod of interest)
        target: string descriptor of class of interest. e.g. "cs" for Computer Science or "math" for 
            Mathematics
    """
    def __init__(self, target: str="cs") -> None:
        super().__init__()
        self.target=target

    def execute(self, input: DataFrame):
        if self.target is None:
            warnings.warn("No filter target selected, returning full dataframe with plaintext labels.")
            output = input
        if self.target is not None:
            print(f"Filtering subset to only contain papers related to '{self.target}' ...")
            # categories for every paper is a list of categories
            cats = input["categories"].tolist()
            cats = [cat[0].split(" ") for cat in tqdm(cats)]
            mask = []
            print("Filtering by majority vote...")
            for item in tqdm(cats): 
                count = 0
                for label in item:
                    if label.startswith(f"{self.target}"):
                        count += 1
                if count >= (len(item)/2):
                    mask.append(True)
                else:
                    mask.append(False)
            # filter subset with majority mask
            output = input.loc[mask]
        return output
    

class ArxivPlaintextLabelStep(Step):
    """
    Optional step that will relabel and add plaintext labels from the arxiv taxonomy loaded in `load_taxonomy` for 
    (Semi-) supervised BERTopic.
    Args:
        input: DataFrame
        threshold: limit for prior label classes if set to value > 0 will filter out papers with very rare label combinations
        taxonomy: Dictionary that maps from l1_labels to plaintext labels. Provided by sciencenow.core.dataset.ArxivDataset
    """
    def __init__(self, taxonomy:Dict[str, str], threshold:int, target:str) -> None:
        super().__init__()
        self.taxonomy = taxonomy
        self.threshold = threshold
        self.target = target

    def execute(self, input: DataFrame) -> DataFrame:
        l1_labels = input["categories"].to_list()
        l1_hardlabels = []
        # only keep labels relevant to computer science
        for item in tqdm(l1_labels):
            temp = item[0].split(" ")
            temp = list(filter(lambda a: a.startswith(f"{self.target}") if self.target is not None else a, temp)) # add fix for no target
            temp = " ".join(temp)
            l1_hardlabels.append(temp)

        input = input.assign(l1_labels=l1_hardlabels)

        if not self.threshold == 0:
            # remove papers that fall into a group with counts less than threshold
            # get a map to obtain label counts to eliminate tiny label groups
            counter_map = Counter(l1_hardlabels)
            hardlabel_counts = [counter_map[label] for label in l1_hardlabels]
            # add textlabels and their set counts to the dataset
            input = input.assign(l1_counts=hardlabel_counts)
            countmask = [count > self.threshold for count in hardlabel_counts]
            input = input.loc[countmask]

        if self.target is not None:
            keys = [key for key in self.taxonomy.keys() if key.startswith(f"{self.target}")]
            target_taxonomy = {key:self.taxonomy[key] for key in keys}
        else:
            target_taxonomy = self.taxonomy

        # convert labels to plaintext
        l1_labels = input["l1_labels"].to_list()
        # obtain plaintext labels
        plaintext_labels = get_plaintext_name(l1_labels, taxonomy=target_taxonomy)
        output = input.assign(plaintext_labels=plaintext_labels)
        return output
    

class ArxivReduceSubsetStep(Step):
    """"
    Optional Step that randomly samples rows from dataframe until target size is reached. 
    Used to avoid out of memory problems while calulating coherence measures during evaluation.
    
    Args:
        limit: integer that specifies the maximum number of papers to reduce the input to
    """
    def __init__(self, limit:int) -> None:
        super().__init__()
        self.limit = limit
    
    def execute(self, input: DataFrame) -> DataFrame:
        if len(input) < self.limit:
            print(f"Input of length {len(input)} already smaller than specified limit.")
            return input
        else:
            output = input.sample(n=self.limit, replace=False)
            return output.sort_values("v1_datetime")


class ArxivGetNumericLabelsStep(Step):
    """
    Obtain a list of numeric labels corresponding to the plaintext labels of the input.
        Used for semisupervised models.

        Args:
            input: DataFrame that contains documents with their respective plaintext labels
            mask_probability:  `float` that specifies the proportion of labels to be masked as -1

        Returns: 
            Tuple(plaintext_labels, numeric_labels)
    """
    def __init__(self, mask_probability: float=0) -> None:
        super().__init__()
        self.mask_probability = mask_probability
    
    def execute(self, input: DataFrame) -> DataFrame:
        if not "plaintext_labels" in input.keys():
            raise NotImplementedError("No reference plaintext labels found. Execute sciencenow.core.steps.ArxivPlaintextLabelStep first.")
        plaintext_labels = input["plaintext_labels"]
        plaintext_map = {k:v for k, v in zip(set(plaintext_labels), list(range(0, len(set(plaintext_labels)), 1)))}
        numeric_labels = [plaintext_map[k] for k in plaintext_labels]
        # Masking a sample of labels with -1, has an impact on supervised UMAP
        if self.mask_probability > 0:
            num_labs = array(numeric_labels)
            mask = array(
                choices(
                    [True, False], 
                    [self.mask_probability, 1-self.mask_probability], 
                    k=len(numeric_labels)
                )
            )
            num_labs[mask] = -1
            numeric_labels = num_labs.tolist()
        output = input.assign(numeric_labels=numeric_labels)
        return output

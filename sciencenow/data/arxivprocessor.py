"""
Wrapper for mass import of Arxiv Data
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from dateutil import parser
from collections import Counter 
import warnings
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer
# from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

from sciencenow.config import (
    FP, 
    PARAMS,
    setup_params,
)

from sciencenow.data.synthdata import embedder
# Make version of UMAP dependent to config so we can choose the correct version depending on the deployment 

if setup_params["recompute"]:
    from cuml.manifold import UMAP # need the GPU implementation to process 2 million embeddings
elif not setup_params["recompute"]:
    from umap import UMAP

# run this in ScienceNOW directory

LABEL_MAP = {
    "stat": "Statistics",
    "q-fin": "Quantitative Finance",
    "q-bio": "Quantitative Biology",
    "cs": "Computer Science",
    "math": "Mathematics"
} # anything else is physics


class ArxivProcessor:
    """
    Wrapper Class that unifies all preprocessing & filtering from the arxiv json data dump.
    The ArxivProcessor needs to be initialized with `FP` and `Params`, two Enum objects that parse the 
    config values saved in `sciencenow.config.secrets.yaml` to tell the ArxivProcessor where the Arxiv Snapshot is located, 
    and which parameters should be used to perform document embedding and UMAP steps.

    Attributes:
        FP: `enum.Enum` that contains all file paths needed to read and write data from disk.
        PARAMS: `enum.Enum` that contains information about which sentence model to use and how to set up the UMAP class for dimensionality reduction.
        label_map: `Dict` as specified above, that maps from Arxiv taxonomy labels to plain text labels.
        arxiv_df: `pandas.DataFrame` that is used as a cache to store all relevant documents.
        sort_by_date: `Bool` that specifies if arxiv_df should be sorted by date in ascending order.
        embeddings: `numpy.array` that stores the raw sentence embeddings obtained for each document.
        reduced_embeddings: `numpy.array` that stores embeddings that have been reduced with UMAP to a lower dimensionality.
        taxonomy: `Dict` that maps from Arxiv taxonomy labels to plain text labels (these are more fine grained, math.PR: Probability instead of just math)
        subset_embeddings: `numpy.array` that serves as cache to store only embeddings relevant to the current subset of interest.
        subset_reduced_embeddings: `numpy.array` that serves as cache to store only reduced embeddings relevant to current subset of interest.
    """
    def __init__(self, sort_by_date=True) -> None:
        """
        Initialize ArxivProcessor object.
        
        Args:
            arxiv_path: `Path` to arxiv OAI snapshot.json file obtained in populate_database.py
            sort_by_date: `Bool` that specifies if data should be sorted by date or not.
        """
        super().__init__()
        self.FP = FP
        self.PARAMS = PARAMS
        self.label_map = LABEL_MAP
        self.arxiv_df = None
        self.sort_by_date = sort_by_date
        self.embeddings = None
        self.reduced_embeddings = None
        self.taxonomy = None
        self.subset_embeddings = None
        self.subset_reduced_embeddings = None
        self.embedder = None

    def load_snapshot(self) -> None:
        """
        Method that reads OAI snapshot into dataframe for easier handling.
        Will check if a .feather file exists and if so, load that since it is both more memory efficient and much faster overall.
        If no .feather file with a preprocessed data frame exists this method will instead load the OAI snapshot.json 
        parse it into a data frame for easier filtering and perform basic preprocessing before serializing to disk.
        """
        if self.FP.FEATHER.value.exists():
            # If the snapshot has already been preprocessed to a .feather file we can just read from disk.
            print(f"Found preprocessed data at {self.FP.FEATHER.value}. Loading from there...")
            self.load_feather()
        else:
            # If we only have the OAI snapshot we remove duplicates, parse date values and serialize to disk.
            print("Loading OAI json snapshot from scratch, this may take a moment...")
            self.arxiv_df = pd.read_json(self.FP.SNAPSHOT.value, lines=True)
            self.arxiv_df = self.arxiv_df.drop_duplicates(subset="id")
            print("Setup complete, parsing datetimes...")
            self.parse_datetime()
            self.preprocess_abstracts()
            print("Serializing data to .feather...}")
            # need to reset index 
            self.arxiv_df = self.arxiv_df.reset_index()
            self.save_feather()

    def save_feather(self):
        """
        Method to store a preprocessed dataframe as .feather file.
        A dataframe of 2 Million documents will take up about 1.9 GB of space on disk.
        """
        self.arxiv_df.to_feather(self.FP.FEATHER.value)
        print(f"Stored dataframe at {self.FP.FEATHER.value}.")

    def load_feather(self):
        """
        Method to load a preprocessed dataframe stored as .feather format.
        """
        self.arxiv_df = pd.read_feather(self.FP.FEATHER.value)
        print(f"Loaded dataframe from {self.FP.FEATHER.value}")

    def parse_datetime(self) -> None:
        """
        Method that converts raw version dates to single publication timestamps for further processing.
        Main idea: Because papers get updated over time, they have more than one time stamp. While the initial v1 timestamp may only mark the initial release and not 
        the official publication in a journal it is still much more logical to use v1 as a proxy for release than to double check every preprint.
        """
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
        else:
            # the versions_dates column contains all update dates in a list for every preprint
            versions = self.arxiv_df.versions_dates
            assert all(versions)
            v1_dates = [x[0] for x in versions]
            print(f"Parsing {len(v1_dates)} version 1 strings to datetime format...")
            # parse to date time format for easier filtering and sorting
            v1_datetimes = [parser.parse(dt) for dt in tqdm(v1_dates)]
            # insert parsed v1_dates as new column
            self.arxiv_df["v1_datetime"] = v1_datetimes
            if self.sort_by_date:
                self.arxiv_df = self.arxiv_df.sort_values("v1_datetime")
            print("Datetime parsing complete.")

    def filter_by_date_range(self, startdate, enddate):
        """
        Returns a subset of the data based on a given start and end date time.
        Will exclusively filter data from 00:00:00 GMT of startdate to 23:59:00 GMT of enddate.

        Args:
            startdate: string of the form "day month year" with their respective numeric values
            enddate: string of the form "day month year" with their respective numeric values
                each seperated by spaces. Example: "31 12 2020" corresponds to the 31st of December 2020.
        """
        if startdate is None:
            warnings.warn("No date range selected, returning full dataframe.")
            return(self.arxiv_df)
        a, b, c = startdate.split(" ")
        # TODO: Why does this only work if we swap b and a for start and end?
        # this actually caused the out of memory errors and other downstream problems, the swap is needed.
        start = pd.to_datetime(parser.parse(f"{b} {a} {c} 00:00:01 GMT"))
        a, b, c = enddate.split(" ")
        end = pd.to_datetime(parser.parse(f"{a} {b} {c} 23:59:59 GMT"))
        # computing mask takes a moment if the dataframe has lots of rows
        mask = [(date > start) & (date < end) for date in tqdm(self.arxiv_df["v1_datetime"])]
        subset = self.arxiv_df.loc[mask]
        return(subset)

    def preprocess_abstracts(self):
        """
        Runs basic preprocessing on abstracts such as removal of newline characters.
        """
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
        else:
            # remove newline characters and strip leading and traling spaces.
            docs = [doc.replace("\n", " ").strip() for doc in self.arxiv_df["abstract"].tolist()]
            self.arxiv_df["abstract"] = docs
            print(f"Successfully removed newlines, leading, and traling spaces from {len(docs)} abstracts.")
    
    @staticmethod
    def extract_vocabulary(subset, fp):
        """
        Extracts a vocabulary from a target subset of data for Topic Model Evaluation with OCTIS
        Args:
            subset: pandas data frame that has been preprocessed and contains all relevant documents in the column "abstract"
            fp: `Path` that specifies location where vocab will be written to.
        """
        docs = subset.abstract.tolist()
        vocab = Counter()
        # Count Vectorizer to be compatible to BERTopic
        tokenizer = CountVectorizer().build_tokenizer()
        print(f"Building vocab for {len(docs)} documents...")
        for doc in tqdm(docs):
            vocab.update(tokenizer(doc))
        with open(fp, "w") as file:
            for item in vocab:
                file.write(item+"\n")
        file.close()
        print(f"Successfully built vocab for subset at {fp}.")

    @staticmethod
    def extract_corpus(subset, fp):
        """
        Extracts a corpus in .tsv format from a target subset of data for Topic Model Evaluation wth OCTIS
        """
        docs = subset.abstract.tolist()
        assert all("\n" not in doc for doc in docs)
        print(f"Building corpus for {len(docs)} documents...")
        with open(fp, "w", encoding='utf-8') as file:
            for document in tqdm(docs):
                # encode-decode to avoid compatibility problems on different platforms
                document = document.encode('utf-8', 'ignore').decode('utf-8')
                file.write(document + "\n")
        file.close()
        print(f"Successfully built corpus at {fp}")

    def load_embeddings(self):
        """
        Loads Embeddings that have been previously saved with `embed_abstracts`
        """
        if not self.FP.EMBEDS.value.exists():
            print(f"No precomputed embeddings found in {self.FP.EMBEDS.value}. Calling `embed_abstracts` first...")
            self.embed_abstracts()
        else:
            self.embeddings = np.load(self.FP.EMBEDS.value)
            print(f"Successfully loaded embeddings for {self.embeddings.shape[0]} documents.")

    def embed_abstracts(self):
        """
        Embeds Abstract texts to vectors with sentence encoder.

        Will skip the SentenceTransformer encoding step if precomputed embeddings can be found on disk.
        Encoding all Abstracts from the Arxiv snapshot on a single A100 40GB GPU will take anywhere from 
        40 to 180 minutes, depending on the pretrained model chosen for the task.
        """
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
            # if embeddings have already been precomputed load them from disk to skip embedding step
        elif self.FP.EMBEDS.value.exists():
            print("Found precomputed embeddings on disk, loading to skip embedding step...")
            self.load_embeddings()
        else:# Embedding from scratch takes about 40 minutes on a 40GB A100
            print(f"No precomputed embeddings on disk, encoding {len(self.arxiv_df)} documents...")
            self.sentence_model = SentenceTransformer(self.PARAMS.SENTENCE_MODEL.value)
            self.embeddings = self.sentence_model.encode(self.arxiv_df["abstract"].tolist(), show_progress_bar=True)
            # saving embeddings to disk
            print(f"Saving embeddings to disk at {self.FP.EMBEDS.value}...")
            np.save(self.FP.EMBEDS.value, self.embeddings, allow_pickle=False)
            # np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=False)
            print(f"Successfully saved {self.embeddings.shape[0]} embeddings to disk.")

    def load_reduced_embeddings(self):
        """
        Loads reduced embeddings that have been previously saved with `reduce_embeddings`
        """
        if not self.FP.REDUCED_EMBEDS.value.exists():
            print(f"No precomputed reduced embeddings found in {self.FP.REDUCED_EMBEDS.value}. Call `reduce_embeddings` first.")
        else:
            self.reduced_embeddings = np.load(self.FP.REDUCED_EMBEDS.value)
            print(f"Successfully loaded reduced embeddings for {self.reduced_embeddings.shape[0]} documents of dimensionality {self.reduced_embeddings.shape[1]}")
        
    def setup_umap_model(self):
        """
        Wrapper to move UMAP setup away from class initialization.
        For (near) real time applications we want to avoid as much computation as possible, thus we only set up the UMAP model if absolutely necessary.
        """
        self.umap_model = UMAP(
                            n_neighbors = self.PARAMS.UMAP_NEIGHBORS.value, 
                            n_components=self.PARAMS.UMAP_COMPONENTS.value, 
                            metric=self.PARAMS.UMAP_METRIC.value, 
                            # low_memory=True, # required for millions of documents on CPU
                            random_state=42)

    def reduce_embeddings(self, subset=None, labels=None):
        """
        Obtain Reduced Embeddings with UMAP to save time in Topic Modeling steps.
        Will only reduce the embeddings of the chosen subset, unless no subset exists, then the entire set of 
        embeddings will be reduced. This drastically shortens computation time for smaller subsets.

        Args:
            subset: subset of data that has been filtered according to the desired specifications
            labels: `np.array` of target labels for semi-supervised modeling
        """
        if subset is not None:
            self.setup_umap_model()
            assert self.subset_embeddings is not None
            # TODO: This fails if we limit the number of maximum papers in app
            self.subset_reduced_embeddings = self.umap_model.fit_transform(self.subset_embeddings, y=labels)
            print(f"Successfully reduced embeddings of subset from {self.subset_embeddings.shape} to {self.subset_reduced_embeddings.shape}")
        elif self.embeddings is None:
            warnings.warn("No embeddings loaded yet. Load them from disk or process a dataset with `embed_abstracts`")
        elif self.FP.REDUCED_EMBEDS.value.exists():
            print(f"Found precomputed reduced embeddings on disk. Loading from {self.FP.REDUCED_EMBEDS.value}...")
            self.load_reduced_embeddings()
        else:
            # Compute reduced embeddings for entire dataset.
            self.setup_umap_model()
            self.reduced_embeddings=self.umap_model.fit_transform(self.embeddings)
            np.save(self.FP.REDUCED_EMBEDS.value, self.reduced_embeddings, allow_pickle=False)
            print(f"Successfully saved {self.reduced_embeddings.shape[0]} reduced embeddings of dimension {self.reduced_embeddings.shape[1]} to disk.")

    def load_taxonomy(self):
        """
        Loads Arxiv Taxonomy for semisupervised models.
        Will load the Arxiv Category Taxonomy (https://arxiv.org/category_taxonomy) as a dictionary from disk.
        This provides us with a map from category labels to plaintext and is also used to obtain numeric class labels
        for the semisupervised model. 
        """
        if not self.FP.TAXONOMY.value.exists():
            warnings.warn(f"No taxonomy found in {self.FP.TAXONOMY.value}. Verify Filepath.")
        else:
            with open(self.FP.TAXONOMY.value, "r") as file:
                taxonomy = [line.rstrip() for line in file]
            taxonomy_list = [line.split(":") for line in taxonomy]
            arxiv_taxonomy = {line[0]: line[1].lstrip() for line in taxonomy_list}
            # add missing label to taxonomy
            arxiv_taxonomy["cs.LG"] = "Machine Learning"
            self.taxonomy = arxiv_taxonomy
            print(f"Successfully loaded {len(self.taxonomy)} labels from {self.FP.TAXONOMY.value}")


    def filter_by_taxonomy(self, subset, target="cs", threshold=100):
        """
        Optional step to filter a subset of papers by soft class labels from Arxiv Taxonomy.
        Will filter a dataframe witih "categories" to only contain articles tagged as the desired category.
        Will relabel and add plaintext labels from the arxiv taxonomy loaded in `load_taxonomy` for 
        (Semi-) supervised BERTopic.
        Because every paper has potentially many different labels this function will filter by simple 
        majority vote. If >= 50% of labels for a paper match the specified target string, the paper
        will be kept in the output subset.
        
        Args: 
            subset: dataframe that should be filtered (already filtered by timeperiod of interest)
            threshold: limit for prior label classes if set to value > 0 will filter out papers with very rare label combinations
            target: string descriptor of class of interest. e.g. "cs" for Computer Science or "math" for 
                Mathematics
        """
        if target is None:
            warnings.warn("No filter target selected, returning full dataframe with plaintext labels.")
            # return subset

        self.load_taxonomy()
        if target is not None:
            print(f"Filtering subset to only contain papers related to {target}...")
            # categories for every paper is a list of categories
            cats = subset.categories.tolist()
            cats = [cat[0].split(" ") for cat in tqdm(cats)]
            mask = []
            print("Filtering by majority vote...")
            for item in tqdm(cats): 
                count = 0
                for label in item:
                    if label.startswith(f"{target}"):
                        count += 1
                if count >= (len(item)/2):
                    mask.append(True)
                else:
                    mask.append(False)
            # filter subset with majority mask
            subset_filtered = subset.loc[mask]
        else:
            subset_filtered = subset
        
        l1_labels = subset_filtered["categories"].to_list()
        l1_hardlabels = []
        # only keep labels relevant to computer science
        for item in tqdm(l1_labels):
            temp = item[0].split(" ")
            temp = list(filter(lambda a: a.startswith(f"{target}") if target is not None else a, temp)) # add fix for no target
            temp = " ".join(temp)
            l1_hardlabels.append(temp)
        # get a map to obtain label counts to eliminate tiny label groups
        counter_map = Counter(l1_hardlabels)
        hardlabel_counts = [counter_map[label] for label in l1_hardlabels]
        # add textlabels and their set counts to the dataset
        subset_filtered = subset_filtered.assign(l1_labels=l1_hardlabels)
        subset_filtered = subset_filtered.assign(l1_counts=hardlabel_counts)
        # remove papers that fall into a group with counts less than threshold
        if target is not None:
            keys = [key for key in self.taxonomy.keys() if key.startswith(f"{target}")]
            target_taxonomy = {key:self.taxonomy[key] for key in keys}
        else:
            target_taxonomy = self.taxonomy
        # filter out very small classes
        countmask = [count > threshold for count in hardlabel_counts]
        subset_filtered = subset_filtered.loc[countmask]
        # convert labels to plaintext
        l1_labels = subset_filtered["l1_labels"].to_list()
        # obtain plaintext labels
        def get_plaintext_name(labs, taxonomy):
            """Will join multi class labels to one distinct label with `&`"""
            p_labels = []
            for item in labs:
                tmp = item.split(" ")
                plaintext_labels = [taxonomy[k] if k in taxonomy else "default" for k in tmp]
                plaintext_labels = " & ".join(plaintext_labels)
                p_labels.append(plaintext_labels)
            return p_labels
        plaintext_labels = get_plaintext_name(l1_labels, taxonomy=target_taxonomy)
        subset_filtered = subset_filtered.assign(plaintext_labels= plaintext_labels)
        print(f"Successfully filtered {len(subset)} documents to {len(subset_filtered)} remaining documents.")
        return subset_filtered

    def reduce_subset(self, subset, limit=7500):
        """"
        Method to randomly reduce subset to target size to avoid out of memory problems while
        calulating coherence measures during evaluation
        """
        if len(subset) < limit:
            print("Rows of subset < specified limit.")
            return subset
        else:
            samp = subset.sample(n=limit, replace=False)
            return samp

    def bertopic_setup(self, subset, recompute=False, labels=None, secondary_subset=None):
        """
        Method to add reduced embeddings to a subset of documents.
        Params: 
            subset: `dataframe` of documents with ids matching the row number in `self.embeddings` and `self.reduced_embeddings`
            recompute: `bool` that specifies if reduced embeddings should be recomputed.
            labels: `np.array` of integer labels used for semisupervised UMAP
            secondary_subset: `dataframe` that is used to load embeddings of synthetic data
        """
        # `index` column of subset matches the index column in processor.arxiv_df
        ids = subset.index.tolist()
        # load ids of secondary subset in case it is part of synthetic data
        # if not recompute: 
        # load only reduced embeddings
        if self.embeddings is None:
            if recompute:
                self.load_embeddings()
            elif not recompute:
                if self.reduced_embeddings is None:
                    self.load_reduced_embeddings()
                self.subset_reduced_embeddings = self.reduced_embeddings[ids]
        if self.arxiv_df is None:
            self.load_snapshot()
        # assert self.embeddings.shape[0] == self.arxiv_df.shape[0]
        if recompute:
            self.subset_embeddings = self.embeddings[ids]
            if secondary_subset is not None:
                if "PUBMED" in secondary_subset.categories.tolist():
                    # TODO: Generalize to other datasets
                    secondary_ids = secondary_subset.index.tolist()
                    # we always recompute with synthetic data
                    self.embedder = embedder(df=None)
                    secondary_embeddings = self.embedder.load_embeddings(Path(self.embedder.cfg.PUBMED_EMBEDS))
                    self.secondary_subset_embeddings = secondary_embeddings[secondary_ids]
                    # secondary labels is easy since we already have fixed labels in secondary subset we only need to 
                    # extend the numeric labels by the classes in 
                    # combining arrays only works if the embeddings have the same shape
                    assert self.subset_embeddings.shape[1] == self.secondary_subset_embeddings.shape[1]
                    _, self.secondary_labels = self.get_numeric_labels(secondary_subset, mask_probability=0)

        # need to grant option to recompute reduced embeddings based on subset and labels for supervised models
        if not recompute:
            # we always recompute, this is just for demonstration purposes
            if self.reduced_embeddings is None:
                self.load_reduced_embeddings()
            self.subset_reduced_embeddings = self.reduced_embeddings[ids]
        elif labels is not None:
            if self.secondary_labels is None:
                # if labels are provided we need to recompute either way
                self.reduce_embeddings(subset=subset, labels=labels)
                print("Recomputed reduced embeddings with their respective labels.")
            elif self.secondary_labels:
                secondary_labs = secondary_labels + max(set(labels)) + 1
                merged_labs = np.concatenate((labels, secondary_labs))
                merged_subset = pd.concat([subset, secondary_subset])
                # now combine embedding arrays
                self.subset_reduced_embeddings = np.concatenate((self.subset_embeddings, self.secondary_subset_embeddings))
                # finally calculate reduced embeddings
                self.reduce_embeddings(subset=merged_subset, labels=merged_labs)
        else:
            print("Recomputing reduced embeddings for subset...")
            self.reduce_embeddings(subset=subset)
        # Setting up corpus and vocabulary
        self.extract_corpus(subset=subset, fp=self.FP.CORPUS.value)
        self.extract_vocabulary(subset=subset, fp=self.FP.VOCAB.value)
        print("BERTopic setup complete.")

    @staticmethod
    def filter_by_labelset(subset1, subset2):
        """
        Filter two subsets to only include papers that fall into classes present in both subsets.
        Used mainly to compare supervised models over multiple years.
        """
        labelset1, labelset2 = set(subset1["plaintext_labels"]), set(subset2["plaintext_labels"])
        label_intersection = labelset1.intersection(labelset2)
        def get_mask(subset, labelset):
            return [lab in labelset for lab in subset["plaintext_labels"]]
        mask1, mask2 = get_mask(subset1, label_intersection), get_mask(subset2, label_intersection)
        res1, res2 = subset1.loc[mask1], subset2.loc[mask2]
        return res1, res2

    @staticmethod
    def get_numeric_labels(subset, mask_probability):
        """
        Obtain a list of numeric labels from the plaintext labels of a subset of data.
        Used for semisupervised models.

        Args:
            subset: `dataframe` that contains documents with their respective plaintext labels
            mask_probability:  `float` that specifies the proportion of labels to be masked as -1

        Returns: 
            Tuple(plaintext_labels, numeric_labels)
        """
        plaintext_labels = subset["plaintext_labels"]
        plaintext_map = {k:v for k, v in zip(set(plaintext_labels), list(range(0, len(set(plaintext_labels)), 1)))}
        numeric_map = {v:k for k, v in plaintext_map.items()}
        numeric_labels = [plaintext_map[k] for k in plaintext_labels]
        if mask_probability > 0:
            num_labs = np.array(numeric_labels)
            mask = np.array(random.choices([True, False], 
                                        [mask_probability, 1-mask_probability], 
                                        k=len(numeric_labels)))
            num_labs[mask] = -1
            numeric_labels = num_labs.tolist()
        numeric_labels = np.array(numeric_labels)
        return plaintext_labels, numeric_labels

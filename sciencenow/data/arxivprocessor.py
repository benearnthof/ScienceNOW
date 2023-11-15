"""
Wrapper for mass import of Arxiv Data
"""

import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import pickle
import re
import string
from dateutil import parser
from omegaconf import OmegaConf
import time
from collections import Counter, defaultdict, OrderedDict 
import argparse
import os 
import pytz
import sys
import warnings
from dateutil import parser
from enum import Enum
from sentence_transformers import SentenceTransformer
from umap import UMAP
#from cuml.manifold import UMAP # need the GPU implementation to process 2 million embeddings


# run this in ScienceNOW directory
cfg = Path(os.getcwd()) / "./sciencenow/config/secrets.yaml"
config = OmegaConf.load(cfg)

class FP(Enum):
    SNAPSHOT = Path(config.ARXIV_SNAPSHOT)
    EMBEDS = Path(config.EMBEDDINGS)
    REDUCED_EMBEDS = Path(config.REDUCED_EMBEDDINGS)
    FEATHER = Path(config.FEATHER_PATH)
    TAXONOMY = Path(config.TAXONOMY_PATH)

LABEL_MAP = {
    "stat": "Statistics",
    "q-fin": "Quantitative Finance",
    "q-bio": "Quantitative Biology",
    "cs": "Computer Science",
    "math": "Mathematics"
} # anything else is physics



subset # 2020 subset
# Extract vocab to be used in BERTopic
from collections import Counter
docs = data
vocab = Counter()
tokenizer = CountVectorizer().build_tokenizer()
for doc in tqdm(docs):
     vocab.update(tokenizer(doc))

with open(custom_path / "vocab.txt", "w") as file:
    for item in vocab:
        file.write(item+"\n")
file.close()

docs = [doc.replace("\n", " ") for doc in docs]
assert all("\n" not in doc for doc in docs)

with open(custom_path / "corpus.tsv", "w") as file:
    for document in docs:
        file.write(document + "\n")
file.close()



class ArxivProcessor:
    """
    Wrapper Class that unifies all preprocessing & filtering from the arxiv json data dump.
    Also able to download papers directly via the arxiv OAI api given a start and end date.
    Params:
        arxiv_path: `Path` to arxiv OAI snapshot.json file obtained in populate_database.py
        sort_by_date: `Bool` that specifies if data should be sorted by date or not.
    """
    def __init__(
        self, 
        sort_by_date=True,
        neighbors=15,
        components=5,
        metric="cosine",
        ) -> None:
        super().__init__()
        self.FP = FP
        self.label_map = LABEL_MAP
        self.arxiv_df = None
        self.sort_by_date = sort_by_date
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2") # TODO: move setup to embedding function
        self.embeddings = None
        self.neighbors = neighbors
        self.components = components
        self.metric="cosine"
        self.umap_model = UMAP(
                            n_neighbors = self.neighbors, 
                            n_components=self.components, 
                            metric=self.metric, 
                            # low_memory=True, # required for millions of documents on CPU
                            random_state=42) # TODO: move setup to embedding function
        self.reduced_embeddings = None
        self.taxonomy = None

    def load_snapshot(self) -> None:
        """Method that reads OAI snapshot into dataframe for easier handling."""
        if self.FP.FEATHER.value.exists():
            print(f"Found preprocessed data at {self.FP.FEATHER.value}. Loading from there...")
            self.load_feather()
        else:
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

    def save_feather(self):S
        """
        Method to store a preprocessed dataframe as .feather file.
        A dataframe of 2 Million documents will take up about 1.9 GB of space on disk.
        """
        self.arxiv_df.to_feather(self.FP.FEATHER.value)
        print(f"Stored dataframe at {self.FP.FEATHER.value}.")

    def load_feather(self):
        """Method to load a preprocessed dataframe stored as .feather format."""
        self.arxiv_df = pd.read_feather(self.FP.FEATHER.value)
        print(f"Loaded dataframe from {self.FP.FEATHER.value}")

    def parse_datetime(self) -> None:
        """Method that converts raw version dates to single publication timestamps for further processing."""
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
        else:
            versions = self.arxiv_df.versions_dates
            assert all(versions) # TODO: move to tests
            v1_dates = [x[0] for x in versions]
            print(f"Parsing {len(v1_dates)} version 1 strings to datetime format...")
            v1_datetimes = [parser.parse(dt) for dt in tqdm(v1_dates)]
            self.arxiv_df["v1_datetime"] = v1_datetimes
            if self.sort_by_date:
                self.arxiv_df = self.arxiv_df.sort_values("v1_datetime")
            print("Datetime parsing complete.")

    def filter_by_date_range(self, startdate, enddate):
        """
        Returns a subset of the data based on a given start and end date time.
        Will exclusively filter data from 00:00:00 GMT of startdate to 23:59:00 GMT of enddate.
        Params:
            startdate: string of the form "day month year" with their respective numeric values
            enddate: string of the form "day month year" with their respective numeric values
                each seperated by spaces. Example: "31 12 2020" corresponds to the 31st of December 2020.
        """
        a, b, c = startdate.split(" ")
        start = pd.to_datetime(parser.parse(f"{a} {b} {c} 00:00:00 GMT"))
        a, b, c = enddate.split(" ")
        end = pd.to_datetime(parser.parse(f"{a} {b} {c} 23:59:59 GMT"))
        # computing mask takes a moment if the dataframe has lots of rows
        mask = [(date > start) & (date < end) for date in tqdm(self.arxiv_df["v1_datetime"])]
        subset = self.arxiv_df.loc[mask]
        return(subset)

    def preprocess_abstracts(self):
        """Runs basic preprocessing on abstracts such as removal of newline characters"""
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
        else:
            # remove newline characters and strip leading and traling spaces.
            docs = [doc.replace("\n", " ").strip() for doc in self.arxiv_df["abstract"].tolist()]
            self.arxiv_df["abstract"] = docs
            print(f"Successfully removed newlines, leading, and traling spaces from {len(docs)} abstracts.")
    
    def extract_vocabulary():
        """Extracts a vocabulary from `self.arxiv_df` for Topic Model Evaluation with OCTIS"""
        pass

    def extract_corpus():
        """Extracts a corpus in .tsv format from `self.arxiv_df` for Topic Model Evaluation wth OCTIS"""
        pass

    def load_embeddings(self):
        """Loads Embeddings that have been previously saved with `embed_abstracts`"""
        if not self.FP.EMBEDS.value.exists():
            print(f"No precomputed embeddings found in {self.FP.EMBEDS.value}. Call `embed_abstracts` first.")
        else:
            self.embeddings = np.load(self.FP.EMBEDS.value())
            print(f"Successfully loaded embeddings for {self.embeddings.shape[0]} documents.")

    def embed_abstracts(self):
        """Embeds Abstract texts to vectors with sentence encoder."""
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
            # if embeddings have already been precomputed load them from disk to skip embedding step
        elif self.FP.EMBEDS.value.exists():
            print("Found precomputed embeddings on disk, loading to skip embedding step...")
            self.load_embeddings()
        else:# Embedding from scratch takes about 40 minutes on a 40GB A100
            print(f"No precomputed embeddings on disk, encoding {len(self.arxiv_df)} documents...")
            self.embeddings = self.sentence_model.encode(self.arxiv_df["abstract"].tolist(), show_progress_bar=True)
            # embeddings = sentence_model.encode(new["abstract"].tolist(), show_progress_bar=True)
            # saving embeddings to disk
            if self.FP.EMBEDS.value.exists():
                print(f"Saving embeddings to disk at {self.FP.EMBEDS.value}...")
                np.save(self.FP.EMBEDS.value, self.embeddings, allow_pickle=False)
                # np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=False)
                print(f"Successfully saved {self.embeddings.shape[0]} embeddings to disk.")

    def load_reduced_embeddings(self):
        """Loads reduced embeddings that have been previously saved with `reduce_embeddings`"""
        if not self.FP.REDUCED_EMBEDS.value.exists():
            print(f"No precomputed reduced embeddings found in {self.FP.REDUCED_EMBEDS.value}. Call `reduce_embeddings` first.")
        else:
            self.reduced_embeddings = np.load(self.FP.REDUCED_EMBEDS.value())
            print(f"Successfully loaded reduced embeddings for {self.reduced_embeddings.shape[0]} documents of dimensionality {self.reduced_embeddings.shape[1]}")
        
    def reduce_embeddings():
        """Obtain Reduced Embeddings with UMAP to save time in Topic Modeling steps."""
        if self.embeddings is None:
            warnings.warn("No embeddings loaded yet. Load them from disk or process a dataset with `embed_abstracts`")
        elif self.FP.REDUCED_EMBEDS.value.exists():
            print(f"Found precomputed reduced embeddings on disk. Loading from {self.FP.REDUCED_EMBEDS.value}...")
            self.load_reduced_embeddings()
        else:
            self.reduced_embeddings=self.umap_model.fit_transform(self.embeddings)
            # reduced_embeddings = umap_model.fit_transform(embeddings)^
            np.save(self.FP.REDUCED_EMBEDS.value, self.reduced_embeddings, allow_pickle=False)
            #np.save(REDUCED_EMBEDDINGS_PATH, reduced_embeddings, allow_pickle=False)
            print(f"Successfully saved {self.reduced_embeddings.shape[0]} reduced embeddings of dimension {self.reduced_embeddings.shape[1]} to disk.")

    def load_taxonomy(self):
        """Loads Arxiv Taxonomy for semisupervised models."""
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
            # keys = arxiv_taxonomy.keys()
            # values = arxiv_taxonomy.values()


    def filter_by_taxonomy(self, subset, target="cs", threshold=100):
        """
        Optional step to filter a subset of papers by soft class labels from Arxiv Taxonomy.
        Will filter a dataframe witih "categories" to only contain articles tagged as the desired category.
        Will relabel and add plaintext labels from the arxiv taxonomy loaded in `load_taxonomy` for 
        (Semi-) supervised BERTopic.
        Because every paper has potentially many different labels this function will filter by simple 
        majority vote. If >= 50% of labels for a paper match the specified target string, the paper
        will be kept in the output subset.
        Params: 
            subset: dataframe that should be filtered (already filtered by timeperiod of interest)
            threshold: limit for prior label classes if set to value > 0 will filter out papers with very rare label combinations
            target: string descriptor of class of interest. e.g. "cs" for Computer Science or "math" for 
                Mathematics
        """
        print(f"Filtering subset to only contain papers related to {target}...")
        # categories for every paper is a list of categories
        cats = subset.categories.tolist()
        cats = [cat[0].split(" ") for cat in tqdm(cats)]
        mask = []
        print("Filtering by majority vote...")
        for item in tqdm(cats): 
            count = 0
            for label in item:
                if label.startswith(f"{target}."):
                    count += 1
            if count >= (len(item)/2):
                mask.append(True)
            else:
                mask.append(False)
        # filter subset with majority mask
        subset_filtered = subset.loc[mask]
        
        l1_labels = subset_filtered["categories"].to_list()
        l1_hardlabels = []
        # only keep labels relevant to computer science
        for item in tqdm(l1_labels):
            temp = item[0].split(" ")
            temp = list(filter(lambda a: a.startswith(f"{target}."), temp))
            temp = " ".join(temp)
            l1_hardlabels.append(temp)
        # get a map to obtain label counts to eliminate tiny label groups
        counter_map = Counter(l1_hardlabels)
        hardlabel_counts = [counter_map[label] for label in l1_hardlabels]
        # add textlabels and their set counts to the dataset
        subset_filtered = subset_filtered.assign(l1_labels=l1_hardlabels)
        subset_filtered = subset_filtered.assign(l1_counts=hardlabel_counts)
        # remove papers that fall into a group with counts less than threshold
        keys = [key for key in self.taxonomy.keys() if key.startswith(f"{target}.")]
        target_taxonomy = {key:self.taxonomy[key] for key in keys}
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
                plaintext_labels = [taxonomy[k] for k in tmp]
                plaintext_labels = " & ".join(plaintext_labels)
                p_labels.append(plaintext_labels)
            return p_labels
        plaintext_labels = get_plaintext_name(l1_labels, taxonomy=target_taxonomy)
        subset_filtered = subset_filtered.assign(plaintext_labels= plaintext_labels)
        print(f"Successfully filtered {len(subset)} documents to {len(subset_filtered)} remaining documents.")
        return subset_filtered

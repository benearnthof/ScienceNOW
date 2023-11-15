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
# import datetime as dt
import sys
import warnings
from dateutil import parser
#from sentence_transformers import SentenceTransformer
#from cuml.manifold import UMAP # need the GPU implementation to process 2 million embeddings


# run this in ScienceNOW directory
cfg = Path(os.getcwd()) / "./sciencenow/config/secrets.yaml"
config = OmegaConf.load(cfg)
ARXIV_PATH = Path(config.ARXIV_SNAPSHOT)
EMBEDDINGS_PATH = Path(config.EMBEDDINGS)
REDUCED_EMBEDDINGS_PATH = Path(config.REDUCED_EMBEDDINGS)
FEATHER_PATH = Path(config.FEATHER_PATH)


embeddings = np.load(EMBEDDINGS_PATH)
embeddings.shape

reduced_embeddings = np.load(REDUCED_EMBEDDINGS_PATH)
reduced_embeddings.shape

arxiv_df = pd.read_json(ARXIV_PATH, lines=True)
arxiv_df = arxiv_df.drop_duplicates(subset="id")

versions = arxiv_df.versions_dates
assert all(versions) # TODO: move to tests
v1_dates = [x[0] for x in versions]
print(f"Parsing {len(v1_dates)} version 1 strings to datetime format...")
v1_datetimes = [parser.parse(dt) for dt in tqdm(v1_dates)]
arxiv_df["v1_datetime"] = v1_datetimes
arxiv_df = arxiv_df.sort_values("v1_datetime")
print("Datetime parsing complete.")
arxiv_df = arxiv_df.reset_index()

arxiv_df.to_feather(FEATHER_PATH)   # 1.849.153
                                    # 1.794.757 after removing trailing spaces and newline characters

# arxiv_df = pd.read_feather(FEATHER_PATH)
start = "1 1 2020"
a, b, c = start.split(" ")
start = pd.to_datetime(parser.parse(f"{a} {b} {c} 00:00:00 GMT"))
end = "31 12 2020"
a, b, c = end.split(" ")
end = pd.to_datetime(parser.parse(f"{a} {b} {c} 23:59:59 GMT"))

mask = [(date > start) & (date < end) for date in tqdm(arxiv_df["v1_datetime"])]

subset = arxiv_df.loc[mask]




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
        arxiv_path=ARXIV_PATH, 
        sort_by_date=True,
        embeddings_path=EMBEDDINGS_PATH,
        reduced_embeddings_path=REDUCED_EMBEDDINGS_PATH,
        neighbors=15,
        components=5,
        metric="cosine",
        feather_path=FEATHER_PATH
        ) -> None:
        super().__init__()
        self.arxiv_path = arxiv_path
        self.arxiv_df = None
        self.sort_by_date = sort_by_date
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.neighbors = neighbors
        self.components = components
        self.metric="cosine"
        self.umap_model = UMAP(
                            n_neighbors = self.neighbors, 
                            n_components=self.components, 
                            metric=self.metric, 
                            low_memory=True, # required for millions of documents
                            random_state=42)
        self.reduced_embeddings_path = reduced_embeddings_path
        self.reduced_embeddings = None
        self.feather_path = feather_path

    def load_snapshot(self) -> None:
        """Method that reads OAI snapshot into dataframe for easier handling."""
        if self.feather_path.exists():
            print(f"Found preprocessed data at {self.feather_path}. Loading from there...")
            self.load_feather()
        else:
            print("Loading OAI json snapshot from scratch, this may take a moment...")
            self.arxiv_df = pd.read_json(self.arxiv_path, lines=True)
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
        self.arxiv_df.to_feather(self.feather_path)
        print(f"Stored dataframe at {self.feather_path}.")

    def load_feather(self):
        """Method to load a preprocessed dataframe stored as .feather format."""
        self.arxiv_df = pd.read_feather(self.feather_path)
        print(f"Loaded dataframe from {self.feather_path}")

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

    def extract_corputs():
        """Extracts a corpus in .tsv format from `self.arxiv_df` for Topic Model Evaluation wth OCTIS"""
        pass

    def load_embeddings(self):
        """Loads Embeddings that have been previously saved with `embed_abstracts`"""
        if not self.embeddings_path.exists():
            print(f"No precomputed embeddings found in {self.embeddings_path}. Call `embed_abstracts` first.")
        else:
            self.embeddings = np.load(self.embeddings_path())
            print(f"Successfully loaded embeddings for {self.embeddings.shape[0]} documents.")

    def embed_abstracts(self):
        """Embeds Abstract texts to vectors with sentence encoder."""
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
            # if embeddings have already been precomputed load them from disk to skip embedding step
        elif self.embeddings_path.exists():
            print("Found precomputed embeddings on disk, loading to skip embedding step...")
            self.load_embeddings()
        else:# Embedding from scratch takes about 40 minutes on a 40GB A100
            print(f"No precomputed embeddings on disk, encoding {len(self.arxiv_df)} documents...")
            self.embeddings = self.sentence_model.encode(self.arxiv_df["abstract"].tolist(), show_progress_bar=True)
            # embeddings = sentence_model.encode(new["abstract"].tolist(), show_progress_bar=True)
            # saving embeddings to disk
            if self.embeddings_path.exists():
                print(f"Saving embeddings to disk at {self.embeddings_path}...")
                np.save(self.embeddings_path, self.embeddings, allow_pickle=False)
                # np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=False)
                print(f"Successfully saved {self.embeddings.shape[0]} embeddings to disk.")

    def load_reduced_embeddings(self):
        """Loads reduced embeddings that have been previously saved with `reduce_embeddings`"""
        if not self.reduced_embeddings_path.exists():
            print(f"No precomputed reduced embeddings found in {self.reduced_embeddings_path}. Call `reduce_embeddings` first.")
        else:
            self.reduced_embeddings = np.load(self.reduced_embeddings_path())
            print(f"Successfully loaded reduced embeddings for {self.reduced_embeddings.shape[0]} documents of dimensionality {self.reduced_embeddings.shape[1]}")
        
    def reduce_embeddings():
        """Obtain Reduced Embeddings with UMAP to save time in Topic Modeling steps."""
        if self.embeddings is None:
            warnings.warn("No embeddings loaded yet. Load them from disk or process a dataset with `embed_abstracts`")
        elif self.reduced_embeddings_path.exists():
            print(f"Found precomputed reduced embeddings on disk. Loading from {self.reduced_embeddings_path}...")
            self.load_reduced_embeddings()
        else:
            self.reduced_embeddings=self.umap_model.fit_transform(self.embeddings)
            # reduced_embeddings = umap_model.fit_transform(embeddings)^
            np.save(self.reduced_embeddings_path, self.reduced_embeddings, allow_pickle=False)
            #np.save(REDUCED_EMBEDDINGS_PATH, reduced_embeddings, allow_pickle=False)
            print(f"Successfully saved {self.reduced_embeddings.shape[0]} reduced embeddings of dimension {self.reduced_embeddings.shape[1]} to disk.")

    def load_taxonomy():
        """Loads Arxiv Taxonomy for semisupervised models."""
        pass

    def filter_by_taxonomy():
        """Optional step to filter a subset of papers by soft class labels from Arxiv Taxonomy"""


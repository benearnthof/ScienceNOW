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
import xml.etree.ElementTree as ET
import argparse
import urllib.request
import urllib
from bs4 import BeautifulSoup
import os 
import pytz
import datetime as dt
import sys
import warnings
from dateutil import parser
from sentence_transformers import SentenceTransformer
# from umap import UMAP
from cuml.manifold import UMAP


cfg = Path(os.getcwd()) / "ScienceNOW/sciencenow/config/secrets.yaml"
config = OmegaConf.load(cfg)
ARXIV_PATH = Path(config.ARXIV_SNAPSHOT)
EMBEDDINGS_PATH = Path(config.EMBEDDINGS)
REDUCED_EMBEDDINGS_PATH = Path(config.REDUCED_EMBEDDINGS)

embeddings = np.load(EMBEDDINGS_PATH)
embeddings.shape

# from cuml.manifold import UMAP
umap_model = UMAP(
    n_components=5, n_neighbors=15, random_state=42, metric="cosine", verbose=True, low_memory=True,
)
reduced_embeddings = umap_model.fit_transform(embeddings)


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
        metric="cosine"
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

    def load_snapshot(self) -> None:
        """Method that reads OAI snapshot into dataframe for easier handling."""
        print("Loading OAI json snapshot from scratch, this may take a moment...")
        self.arxiv_df = pd.read_json(self.arxiv_path, lines=True)
        self.arxiv_df = self.arxiv_df.drop_duplicates(subset="id")
        print("Setup complete.")

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
        """Returns a subset of the data based on a given start and end date time"""
        pass

    def preprocess_abstracts(self):
        """Runs basic preprocessing on abstracts such as removal of newline characters"""
        if self.arxiv_df is None:
            warnings.warn("No data loaded yet. Load snapshot or fully processed dataset first.")
        else:
            # remove newline characters and strip leading and traling spaces.
            docs = [doc.replace("\n", " ") for doc in self.arxiv_df["abstract"].tolist()]
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


"""
Preprocessing for Synthetic Trend Auxiliary Datasets
"""

# from sciencenow.data.arxivprocessor import ArxivProcessor
# processor = ArxivProcessor(sort_by_date=True)
# # processor.load_snapshot()

# every document needs an id, title, abstract
# v1_datetime will be adjusted on the fly
# base embeddings need to be calculated of course
# from sciencenow.models.train import ModelWrapper
# from sciencenow.config import (
#     setup_params,
#     online_params
# )

# wrapper = ModelWrapper(setup_params=setup_params, model_type="semisupervised", usecache=True)



# starting with pubmed

from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from math import floor
from sentence_transformers import SentenceTransformer

from sciencenow.config import CONFIGPATH


cfg = OmegaConf.load(CONFIGPATH)

pubmedpath = Path(cfg.PUBMED_PATH)

def preprocess_pubmed(pubmedpath=pubmedpath, target_amount=100):
    """
    Will return a dataframe of pubmed articles with ids, abstracts and labels.
    Requires the pubmed train.txt file.
    """
    if not pubmedpath.exists():
        raise FileNotFoundError("Pubmed file not found.")
    with open(pubmedpath) as file:
        lines = [line.rstrip().split("\t")[1] if "\t" in line else line.rstrip() for line in file]
    chunks = {}
    temp = []
    # In pubmed every abstract is tagged with an ID
    # every abstract is split into lines that consist of a tag and a sentence separated by a tab.
    # we stripped out the tags and tabs and are left with lines of sentences we need to combine 
    for line in lines:
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
    pubmed_df = pd.DataFrame(data)
    pubmed_df = pubmed_df.sample(n = floor(target_amount))
    return pubmed_df

# pubmed_df = preprocess_pubmed()

class embedder():
    """
    Embeds Abstract texts to vectors with sentence encoder. Like
    """
    def __init__(self, df, cfg=cfg, dataset="pubmed") -> None:
        self.df = df
        self.cfg = cfg
        self.dataset=dataset

    def embed_dataset(self):
        if self.dataset not in ["pubmed", "movies", "yahoo"]:
            raise NotImplementedError(f"Not yet implemented for dataset {self.dataset}")

        if self.df is None:
            raise NotImplementedError("Embedder initialized without dataset.")

        if self.dataset == "pubmed":
            if Path(self.cfg.PUBMED_EMBEDS).exists():
                print("Found precomputed embeddings on disk, loading to skip embedding step...")
                embeddings = np.load(Path(self.cfg.PUBMED_EMBEDS))
                print(f"Successfully loaded embeddings for {embeddings.shape[0]} documents.")
                return embeddings
            else:
                self.embed(self.df, path=Path(self.cfg.PUBMED_EMBEDS))

        elif self.dataset == "movies":
            raise NotImplementedError
        elif self.dataset == "yahoo":
            raise NotImplementedError
    
    def embed(self, path):
        print(f"No precomputed embeddings on disk, encoding {len(self.df)} documents...")
        self.sentence_model = SentenceTransformer(self.cfg.SENTENCE_MODEL)
        embeddings = self.sentence_model.encode(self.df["abstract"].tolist(), show_progress_bar=True)
        # saving embeddings to disk
        print(f"Saving embeddings to disk at {path}...")
        np.save(path, embeddings, allow_pickle=False)
        # np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=False)
        print(f"Successfully saved {embeddings.shape[0]} embeddings to disk.")
        return embeddings
    
    def load_embeddings(self, path):
        print(f"Loading Secondary Embeddings from disk...")
        return np.load(path)
    

# need to merge the embeddings with the arxiv_df somehow
# processor.filter_by_taxonomy(subset=pubmed_df, target=None, threshold=0)

# need to adjust processor.subset_embeddings
# currently it is just an array of all embeddings but we need a way to load the embeddings for the synthetic data
# possible solution to memory problems: Incremental UMAP

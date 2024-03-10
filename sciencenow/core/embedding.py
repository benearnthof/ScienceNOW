from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer

from numpy import (
    load as load_array,
    save as save_array,
)

class Embedder(ABC):
    """
    Abstract Base Class for all Embedding functionality.

    Args:
        source: optional string that specifies path where preprocessed embeddings are located.
        target: optional string that specifies path where embeddings should be written to disk.
        data: List of strings that will be encoded to embeddings.
        sentence_model: String descriptor that specifies which pretrained sentence transformer to load.
    """
    source: Optional[str]
    target: Optional[str]
    data: Optional[List[str]]
    sentence_model: str

    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    def embed(self):
        raise NotImplementedError
    

class ArxivEmbedder(Embedder):
    """
    Class that implements embedding functionality for a list of Arxiv embeddings

    Args:
        source: Optional string that specifies location of preprocessed embeddings.
        target: Optional string that specifies where to write embeddings to disk.
        data: List of strings that will be embedded.
    """
    def __init__(
            self, 
            source:Optional[str], 
            target:Optional[str],
            data:Optional[List[str]],
            ) -> None:
        super().__init__()
        self.source=Path(source) if source is not None else None
        self.target=Path(target) if source is not None else None
        self.data=data
        self.embeddings=None
    
    def load(self) -> None: 
        """
        Loads Embeddings that have been previously saved with `self.save`
        """
        if not self.source.exists():
            raise NotImplementedError(f"No precomputed embeddings found in {self.source}. Call `embed` first.")
        self.embeddings = load_array(self.source)
        print(f"Successfully loaded embeddings for {self.embeddings.shape[0]} documents.")

    def save(self) -> None:
        if self.target is None:
            raise NotImplementedError("Unable to save embeddings, no target provided.")
        if self.embeddings is None:
            raise NotImplementedError("Cannot save empty array, call `embed` first.")
        save_array(self.target, self.embeddings, allow_pickle=False)
        print(f"Successfully saved {self.embeddings.shape[0]} embeddings to disk.")

    def embed(self, sentence_model: str = "distilroberta-base") -> None:
        """
        Embeds Abstract texts to vectors with sentence encoder.
        Encoding all Abstracts from the Arxiv snapshot on a single A100 40GB GPU will take anywhere from 
        40 to 180 minutes, depending on the pretrained model chosen for the task.
        """
        if self.data is None:
            raise NotImplementedError("Provide list of strings to embed.")
        else:
            print(f"Encoding {len(self.data)} documents...")
            sentence_model = SentenceTransformer(sentence_model)
            self.embeddings = sentence_model.encode(self.data, show_progress_bar=True)

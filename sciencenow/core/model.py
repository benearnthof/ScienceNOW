from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
from collections import Counter

from pandas import DataFrame
from numpy import array
from tqdm import tqdm

from sciencenow.core.dimensionality import Dimensionality
from sciencenow.core.clustering import River
from sciencenow.core.utils import chunk_list

from river.cluster import DBSTREAM
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer, OnlineCountVectorizer

from sklearn.feature_extraction.text import CountVectorizer


class TopicModel(ABC):
    """
    Abstract Base Class for Topic Models.
    """
    data: DataFrame
    embeddings: array
    reducer: Dimensionality
    cluster_model: HDBSCAN
    ctfidf_model: ClassTfidfTransformer
    corpus: str
    vocab: str

    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        raise NotImplementedError
    

class BERTopicBase(TopicModel):
    """
    Wrapper Class that takes care of all setup steps needed for BERTopic training and evaluation.
    
    Args:
        data: DataFrame containing documents in "abstracts" and timestamps in "v1_timestamps"
        embeddings: array containing reduced sentence embeddings of documents
        reducer: Reducer class that provides dimensionality reduction functionality
        cluster_model: Any cluster model of choice
        ctfidf_model: Any ClassTfidfTransformer
    """
    def __init__(
            self,
            data: DataFrame, 
            embeddings: array,
            cluster_model: Any,
            ctfidf_model: Any,
            ) -> None:
        super().__init__()
        self.data = data
        self.embeddings = embeddings # sciencenow.core.dimensionality.UmapReducer.reduced_embeddings
        self.reducer = Dimensionality(self.embeddings) # embeddings for subset are always already reduced
        self.cluster_model = cluster_model
        self.ctfidf_model = ctfidf_model
        self.vectorizer_model = CountVectorizer(stop_words="english")

        self.topics = None
        self.topic_info = None
        self.corpus = None
        self.vocabulary = None
        
        self.topic_model = BERTopic(
            umap_model=self.reducer,
            hdbscan_model=self.cluster_model,
            ctfidf_model=self.ctfidf_model,
            vectorizer_model=self.vectorizer_model,
            verbose=True,
            calculate_probabilities=False,
        )
        print("Setup complete.")

    def load(self, path: str) -> None:
        """
        Load a pretrained Topic Model from disk.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No topic model found at location: {path}")
        self.topic_model = BERTopic.load(path)
        print(f"Topic model loaded successfully from: {path}")

    def save(self, path: str) -> None:
        """
        Save a trained Topic Model to disk
        """
        path = Path(path)
        if self.topic_model is None:
            raise NotImplementedError("Cannot save empty topic model.")
        
        self.topic_model.save(
            path=path,
            serialization="safetensors",
            save_ctfidf=True
            )
        print(f"Model saved successfully at {path}")

    def extract_corpus(self, path:str) -> None:
        """
        Saves corpus in .tsv format for topic model evaluation with OCTIS.
        Used with `get_dataset` to load dataset in OCTIS format

        Args:
            path: string that specifies desired location of corpus file.
        """
        path = Path(path)
        docs = self.data.abstract.tolist()
        print(f"Building corpus for {len(docs)} documents...")
        with open(path, "w", encoding='utf-8') as file:
            for document in tqdm(docs):
                # encode-decode to avoid compatibility problems on different platforms
                document = document.encode('utf-8', 'ignore').decode('utf-8')
                file.write(document + "\n")
        file.close()
        print(f"Successfully built corpus at {path}")
        self.corpus = path
    
    def extract_vocabulary(self, path: str) -> None:
        """
        Extracts vocabulary from a dataset for Topic Model Evaluation with OCTIS

        Args:
            path: string that specifies location where vocab will be written to.
        """
        path = Path(path)
        docs = self.data.abstract.tolist()
        vocab = Counter()
        tokenizer = CountVectorizer().build_tokenizer()
        print(f"Building vocab for {len(docs)} documents...")
        for doc in tqdm(docs):
            vocab.update(tokenizer(doc))
        with open(path, "w") as file:
            for item in vocab:
                file.write(item+"\n")
        file.close()
        print(f"Successfully built vocab for subset at {path}.")
        self.vocabulary = path

    def train(self):
        """
        Train a basic Topic model. Neither Online nor Dynamic
        """
        if self.topic_model is None:
            print(f"Setting up Topic Model...")
        self.topics, _ = self.topic_model.fit_transform(
            documents=self.data.abstract.tolist(),
            embeddings=self.embeddings
            )# We Never provide labels since labels only influence UMAP and the embeddings are already reduced
        self.topic_info = self.topic_model.get_topic_info()
        print("Base Topic Model fit successfully.")


class BERTopicOnline(BERTopicBase):
    """
    Extends BERTopic Base to online Model. Requires slightly different setup and training.
    """
    def __init__(
            self,
            data: DataFrame,
            embeddings: array,  
            cluster_model: Any,
            cluster_params: Dict[str, Any],
            ctfidf_model: Any
            ) -> None:
        super().__init__(data, embeddings, cluster_model, ctfidf_model)
        self.data = data
        self.embeddings = embeddings # sciencenow.core.dimensionality.UmapReducer.reduced_embeddings
        self.reducer = Dimensionality(self.embeddings) # embeddings for subset are always already reduced
        self.cluster_model = River(model=DBSTREAM(**cluster_params))
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
        self.vectorizer_model = OnlineCountVectorizer(min_df=10, stop_words="english") # min_df to avoid memory problems during eval
        # TODO: Doublecheck if this is needed, should be handled by init of superclass already
        self.topics = None
        self.topic_info = None
        self.topics_over_time = None
        self.corpus = None
        self.vocabulary = None

        self.topic_model = BERTopic(
            umap_model=self.reducer,
            hdbscan_model=self.cluster_model,
            ctfidf_model=self.ctfidf_model,
            vectorizer_model=self.vectorizer_model,
            verbose=True,
            calculate_probabilities=False,
        )
        print("Setup complete.")

    def train(self, setup_params: Dict[str, Any]) -> None:
        """
        Train an Online Topic Model.
        """
        docs = self.data.abstract.tolist()
        doc_chunks = chunk_list(docs, n=setup_params["nr_chunks"])
        timestamps = self.data.v1_datetime.tolist()
        self.topics = []
        for docs in tqdm(doc_chunks):
            self.topic_model.partial_fit(docs)
            self.topics.extend(self.topic_model.topics_)
        # for postprocessing
        self.topic_model.topics_ = self.topics
        # need to quantize into bins for trend extraction
        self.topics_over_time = self.topic_model.topics_over_time(
            docs, 
            timestamps, 
            nr_bins=setup_params["nr_bins"],
            evolution_tuning=setup_params["evolution_tuning"],
            global_tuning=setup_params["global_tuning"],
            )
        self.topic_info = self.topic_model.get_topic_info()


class BERTopicDynamic(BERTopicBase):
    """
    Extends BERTopic Base to dynamic Model. Requires slightly different setup and training.
    """
    def __init__(
            self, 
            data: DataFrame, 
            embeddings: array, 
            cluster_model: Any, 
            ctfidf_model: Any) -> None:
        super().__init__(data, embeddings, cluster_model, ctfidf_model)
        self.data = data
        self.embeddings = embeddings
        self.reducer = Dimensionality(self.embeddings)
        self.cluster_model = cluster_model
        self.vectorizer_model = CountVectorizer(stop_words="english")

        self.topics = None
        self.topic_info = None
        self.topics_over_time = None
        self.corpus = None
        self.vocabulary = None

        self.topic_model = BERTopic(
            umap_model=self.reducer,
            hdbscan_model=self.cluster_model,
            ctfidf_model=self.ctfidf_model,
            vectorizer_model=self.vectorizer_model,
            verbose=True,
            calculate_probabilities=False,
        )
        print("Setup complete.")

    def train(self, setup_params: Dict[str, Any]) -> None:
        """
        Train a dynamic topic model.
        """
        docs = self.data.abstract.tolist()
        embeddings = self.embeddings
        timestamps = self.data.v1_datetime.tolist()
        self.topics, _ = self.topic_model.fit_transform(docs, embeddings)
        # reassign to hopefully help with topicover time representation
        # https://github.com/MaartenGr/BERTopic/issues/1593
        self.topic_model.topics_ = self.topics
        print(f"Fitting dynamic model with {len(timestamps)} timestamps and {setup_params['nr_bins']} bins.")
        self.topics_over_time = self.topic_model.topics_over_time(
            docs,
            timestamps,
            nr_bins=setup_params["nr_bins"],
            evolution_tuning=setup_params["evolution_tuning"],
            global_tuning=setup_params["global_tuning"],
            )
        self.topic_info = self.topic_model.get_topic_info()



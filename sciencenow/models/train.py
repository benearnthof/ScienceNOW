"""
Wrapper class that unifies Topic Model training. 
"""

from pathlib import Path
from omegaconf import OmegaConf
from enum import Enum
from os import getcwd
from collections import Counter
from tqdm import tqdm
import warnings

from sklearn.feature_extraction.text import CountVectorizer
# from sentence_transformers import SentenceTransformer
from cuml.cluster import HDBSCAN
from bertopic import BERTopic


from sciencenow.data.arxivprocessor import ArxivProcessor
from sciencenow.utils.wrappers import Dimensionality, River

processor = ArxivProcessor()

processor.load_snapshot()

startdate = "01 01 2020"
enddate = "31 12 2020"
target = "cs"
threshold = 100

subset = processor.filter_by_date_range(startdate=startdate, enddate=enddate) 
subset = processor.filter_by_taxonomy(subset=subset, target=target, threshold=threshold)

processor.bertopic_setup(subset=subset, recompute=True)
# now the processor class is ready to train topic models
subset_reduced_embeddings = processor.subset_reduced_embeddings

# run this in ScienceNOW directory
cfg = Path(getcwd()) / "./sciencenow/config/secrets.yaml"
config = OmegaConf.load(cfg)

class TM_PARAMS(Enum):
    """
    Wrapper class for Topic Model hyper parameters.
    Params:
        VOCAB_THRESHOLD:`int` that specifies how many times a word must occur in the corpus for 
            it to be contained in the topic model vocabulary. (This is separate from the vocab used for evaluation.)
    """
    VOCAB_THRESHOLD=config.VOCAB_THRESHOLD
    TM_VOCAB_PATH=config.TM_VOCAB_PATH
    TM_TARGET_ROOT=config.TM_TARGET_ROOT

hdbscan_params = {
    "samples": 30,
    "cluster_size": 30,
}

class ModelWrapper():
    """
    Class to unify setup steps for Topic Model (tm) training. 
    Params:
        subset: `dataframe` that contains documents, timestamps and labels
        tm_params: `TM_PARAMS` enum that contains all hyperparameters
        hdbscan_params: `Dict` with hyperparameters for HDBSCAN
        model_type: `str`; one of "base", "dynamic", "online", "antm" 
    """
    def __init__(
        self, 
        subset=None,
        subset_reduced_embeddings=None,
        tm_params=TM_PARAMS,
        hdbscan_params=hdbscan_params,
        model_type="base",

        ) -> None:
        super().__init__()
        self.subset = subset
        self.subset_reduced_embeddings = subset_reduced_embeddings
        self.tm_params = tm_params
        self.tm_vocab = None
        self.hdbscan_params=hdbscan_params
        model_types=["base", "dynamic", "online", "antm"]
        if model_type not in model_types:
            raise ValueError(f"Invalid model type. Expected on of {model_types}")
        self.model_type = model_type
        self.topics = None
        self.probs = None

    def generate_tm_vocabulary(self, recompute=True):
        """
        Generate a vocabulary to offload this computation from tm training.
        Params:
            recompute: `Bool` indicating if vocabulary should be recomputed for subset of interest.
        """
        assert self.subset is not None
        if recompute:
            print(f"Recomputing vocab for {len(self.subset)} documents...")
            vocab = Counter()
            tokenizer = CountVectorizer().build_tokenizer()
            docs = self.subset.abstract.tolist()
            for doc in tqdm(docs):
                vocab.update(tokenizer(doc))
            reduced_vocab = [word for word, freq in vocab.items() if freq >= self.tm_params.VOCAB_THRESHOLD.value]
            print(f"Reduced vocab from {len(vocab)} to {len(reduced_vocab)}")
            with open(self.tm_params.TM_VOCAB_PATH.value, "w") as file:
                for item in reduced_vocab:
                    file.write(item+"\n")
            file.close()
            self.tm_vocab = reduced_vocab
            print(f"Successfully saved Topic Model vocab at {self.tm_params.TM_VOCAB_PATH.value}")
        else:
            self.load_tm_vocabulary()
        
    def load_tm_vocabulary(self):
        """Wrapper to quickly load a precomputed tm vocabulary to avoid recomputing between runs."""
        if not self.tm_params.TM_VOCAB_PATH.value.exists():
            warnings.warn(f"No tm vocabulary found at {self.tm_params.TM_VOCAB_PATH.value}.
                            Consider calling `generate_tm_vocabulary` first.")
        else:
            with open(self.tm_params.TM_VOCAB_PATH.value, "r") as file:
                self.tm_vocab = [row.strip() for row in file]
            file.close()
            print(f"Successfully loaded TopicModel vocab of {len(self.tm_vocab)} items.")

    def tm_setup(self)
        """
        Wrapper to quickly set up a new topic model.
        """
        # use precomputed reduced embeddings 
        self.umap_model = Dimensionality(self.subset_reduced_embeddings)
        self.hdbscan_model = HDBSCAN( # TODO: add KMEANS & River for online & supervised models
            min_samples=self.hdbscan_params["samples"],
            gen_min_span_tree=True,
            prediction_data=True,
            min_cluster_size=self.hdbscan_params["cluster_size"],
            verbose=True,
        )
        # remove stop words for vectorizer just in case
        self.vectorizer_model = CountVectorizer(vocabulary=self.tm_vocab, stop_words="english")
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            #vectorizer_model=self.vectorizer_model, # TODO: Investigate ValueError: Input contains infinity or a value too large for dtype('float64').
            verbose=True,
        )
        print("Setup complete.")

    def tm_train(self):
        """
        Wrapper to train topic model.
        """
        if self.model_type == "base":
            docs = self.subset.abstract.tolist()
            embeddings = self.subset_reduced_embeddings
            self.topics, self.probs = self.topic_model.fit(documents=docs, embeddings=embeddings)
            print("Base Topic Model fit successfully.")
    
    def tm_save(self, name):
        """
        Wrapper to save a trained topic model.
        """
        if self.topic_model is not None:
            self.topic_model.save(
                path=Path(self.tm_params.TM_TARGET_ROOT.value) / name,
                serialization="safetensors",
                save_ctfidf=True
                )
            print(f"Model saved successfully in {self.tm_params.TM_TARGET_ROOT.value}")
        else:
            print(f"No Model found at specified location: {self.tm_params.TM_TARGET_ROOT.value}")

    def tm_load(self, name):
        """
        Wrapper to load a pretrained topic model.
        """
        assert (Path(self.tm_params.TM_TARGET_ROOT.value) / name).exists()
        self.topic_model = BERTopic.load(Path(self.tm_params.TM_TARGET_ROOT.value) / name)
        print(f"Topic model at {Path(self.tm_params.TM_TARGET_ROOT.value) / name}loaded successfully.")

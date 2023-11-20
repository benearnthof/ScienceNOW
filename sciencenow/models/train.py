"""
Wrapper class that unifies Topic Model training. 
"""

from pathlib import Path
from omegaconf import OmegaConf
from enum import Enum
from os import getcwd
from collections import Counter
from tqdm import tqdm
import numpy as np
import random
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from cuml.cluster import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer


from sciencenow.data.arxivprocessor import ArxivProcessor
from sciencenow.utils.wrappers import Dimensionality, River

# TODO: Move this to tests
"""processor = ArxivProcessor()
processor.load_snapshot()
startdate = "01 01 2020"
enddate = "31 12 2020"
target = "cs"
threshold = 100
subset = processor.filter_by_date_range(startdate=startdate, enddate=enddate) 
subset = processor.filter_by_taxonomy(subset=subset, target=target, threshold=threshold)
plaintext_labels, numeric_labels = processor.get_numeric_labels(subset1, mask_probability=0)

processor.bertopic_setup(subset=subset, recompute=True, labels=numeric_labels)
# # now the processor class is ready to train topic models
subset_reduced_embeddings = processor.subset_reduced_embeddings
"""
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

setup_params = {
    "samples": 30, # hdbscan samples
    "cluster_size": 30, # hdbscan minimum cluster size
    "startdate": "01 01 2020", # if no date range should be selected set startdate to `None`
    "enddate":"31 12 2020",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "threshold": 100,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": 0,
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 52 # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
}

class ModelWrapper():
    """
    Class to unify setup steps for Topic Model (tm) training. 
    Params:
        subset: `dataframe` that contains documents, timestamps and labels
        tm_params: `TM_PARAMS` enum that contains all hyperparameters
        setup_params: `Dict` with hyperparameters for model setup
        model_type: `str`; one of "base", "dynamic", "online", "antm" 
    """
    def __init__(
        self, 
        subset_reduced_embeddings=None,
        tm_params=TM_PARAMS,
        setup_params=setup_params,
        model_type="base",
        ) -> None:
        super().__init__()
        #### Setting up data subset, labels & embeddings via processor
        self.processor = ArxivProcessor()
        self.processor.load_snapshot()
        self.tm_params = tm_params
        self.setup_params = setup_params
        self.subset = self.processor.filter_by_date_range(
            startdate=self.setup_params["startdate"],
            enddate=self.setup_params["enddate"]
            )
        self.subset = self.processor.filter_by_taxonomy(
            subset=self.subset, 
            target=self.setup_params["target"], 
            threshold=self.setup_params["threshold"]
            )
        self.plaintext_labels, self.numeric_labels = processor.get_numeric_labels(
            subset = self.subset,
            mask_probability=self.setup_params["mask_probability"])
        model_types=["base", "dynamic", "semisupervised", "online", "antm", "embetter"]
        if model_type not in model_types:
            raise ValueError(f"Invalid model type. Expected on of {model_types}")
        self.model_type = model_type
        if self.model_type == "semisupervised": #recompute embeddings with supervised umap
            self.processor.bertopic_setup(
                subset=self.subset, recompute=True, labels=self.numeric_labels
                )
        else:
            self.processor.bertopic_setup(
                subset=self.subset, recompute=True
                )
        self.subset_reduced_embeddings = self.processor.subset_reduced_embeddings
        #### vocab for evaluation
        self.tm_vocab = None
        #### outputs
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.topic_info = None
        self.topics_over_time = None

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
            min_samples=self.setup_params["samples"],
            gen_min_span_tree=True,
            prediction_data=True,
            min_cluster_size=self.setup_params["cluster_size"],
            verbose=True,
        )
        # remove stop words for vectorizer just in case
        self.vectorizer_model = CountVectorizer(vocabulary=self.tm_vocab, stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            ctfidf_model=self.ctfidf_model,
            #vectorizer_model=self.vectorizer_model, # TODO: Investigate ValueError: Input contains infinity or a value too large for dtype('float64').
            verbose=True,
            nr_topics=self.setup_params["nr_topics"]
        )
        print("Setup complete.")

    def tm_train(self):
        """
        Wrapper to train topic model.
        """
        if self.topic_model is None:
            warnings.warn("No topic model set up yet. Call `tm_setup` first.")
        if self.model_type == "base":
            docs = self.subset.abstract.tolist()
            embeddings = self.subset_reduced_embeddings
            self.topics, self.probs = self.topic_model.fit(documents=docs, embeddings=embeddings)
            self.topic_info = self.topic_model.get_topic_info()
            print("Base Topic Model fit successfully.")
        elif self.model_type in ["dynamic", "semisupervised"]:
            # Training procedure is the same since both models use timestamps and for semisupervised
            # the reduced embeddings were already recomputed based on the numeric labels while initializing
            # the class
            docs = self.subset.abstract.tolist()
            embeddings = self.subset_reduced_embeddings
            timestamps = self.subset.v1_datetime.tolist()
            self.topics, self.probs = self.topic_model.fit_transform(docs, embeddings)
            print(f"Fitting dynamic model with {len(timestamps)} timestamps and {self.setup_params["nr_bins"]} bins.")
            self.topics_over_time = self.topic_model.topics_over_time(
                docs, 
                timestamps, 
                nr_bins=self.setup_params["nr_bins"]
                )
            self.topic_info = self.topic_model.get_topic_info()
        elif self.model_type == "online": # TODO: adapt from script
            pass
        elif self.model_type == "antm": # TODO: adapt from script
            pass
        elif self.model_type == "embetter" # TODO: adapt from script
            pass
    
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


"""
# Short hand to test setup
umap_model = Dimensionality(subset_reduced_embeddings)
hdbscan_model = HDBSCAN( # TODO: add KMEANS & River for online & supervised models
            min_samples=setup_params["samples"],
            gen_min_span_tree=True,
            prediction_data=True,
            min_cluster_size=setup_params["cluster_size"],
            verbose=True,
        )

topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            #vectorizer_model=self.vectorizer_model, # TODO: Investigate ValueError: Input contains infinity or a value too large for dtype('float64').
            verbose=True,
            nr_topics=None
        )
"""



# to test filter by labelset
#startdate_21 = "01 01 2021"
#enddate_21 = "31 12 2021"
#subset_21 = processor.filter_by_date_range(startdate=startdate_21, enddate=enddate_21)
#subset_21 = processor.filter_by_taxonomy(subset=subset_21, target=target, threshold=threshold)
# subset1, subset2 = processor.filter_by_labelset(subset, subset_21)

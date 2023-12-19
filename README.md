# ScienceNOW
ScienceNOW: Topic Modelling with Arxiv e-prints for Trend detection.

Installation: 
TODO: 
  clean up setup (remove all relative file paths & test install with pip)
  clean up dependencies for install

Setting up the project config:  
To successfully train and evaluate topic models with this repository you need to first set up a config file. 
Navigate from the root of the project directory to the folder  
`sciencenow/config/`  
and create the file `secrets.yaml`  
It must be structured in the following way:  
```yaml
ROOT: "absolute path to the Project Root e.g.: ./ScienceNOW/"
ARXIV_SNAPSHOT: "absolute path to the metadata snapshot e.g.: /arxiv-public-datasets/arxiv-data/arxiv-metadata-oai-2023-11-13.json"
EMBEDDINGS: "absolute path to the precomputed abstract sentence embeddings e.g.: /arxiv-public-datasets/arxiv-data/embeddings.npy"
REDUCED_EMBEDDINGS: absolute path to the precomputed reduced embeddings e.g.: /arxiv-public-datasets/arxiv-data/reduced_embeddings.npy"
FEATHER_PATH: "absolute path to the preprocessed metadata in .feather format e.g.:/arxiv-public-datasets/arxiv-data/arxiv_df.feather"
TAXONOMY_PATH: "absolute path to the label taxonomy for semisupervised models e.g.: /taxonomy.txt"
EVAL_ROOT: "absolute path to the directory in which evaluation results should be stored e.g.: /tm_evaluation/"
VOCAB_PATH: "absolute path to the file where the vocabulary for evaluation will be stored e.g.: /tm_evaluation/vocab.txt"
CORPUS_PATH: "absolute path to the file where the corpus used for evaluation will be stored e.g.: /tm_evaluation/corpus.tsv"
SENTENCE_MODEL: "sentence transformer used to generate EMBEDDINGS e.g.: all-MiniLM-L6-v2" 
UMAP_NEIGHBORS: 15 umap parameters
UMAP_COMPONENTS: 5
UMAP_METRIC: "cosine"
VOCAB_THRESHOLD: 15 # Threshold used to avoid out of memory errors during evaluation
TM_VOCAB_PATH: "absolute path to the topic model vocab (distinct from evaluation vocab) e.g.: /tm_evaluation/tm_vocab.txt"
TM_TARGET_ROOT: "absolute path to directory where trained topic model should be written to disk e.g.: /tm_evaluation/"
```

We will go over each of these parameters (and other training hyperparameters) in detail, for now let's set up the data we wish to analyze.  

Setting up the Arxiv Snapshot:  
To do Topic Modelling we first need to download the metadata of all Arxiv preprints.  
To accomplish this we use [https://github.com/mattbierbaum/arxiv-public-datasets/tree/master](https://github.com/mattbierbaum/arxiv-public-datasets/tree/master)  
In combination with the file `sciencenow.data.update_arxiv_snapshot.py`.  
Because downloading the data from the OAI takes more than 6 hours (because the OAI returns a maximum of 1000 records every 10 seconds), we combine predownloaded data 
with new data by utilizing the resumption token obtained while downloading the data for the very first time.  
This looks like we still need to run the download at least one time though?  
Luckily you can download all data up until November of 2023 from here:  
[https://drive.google.com/drive/folders/1xhLDDFwJauVH5ijRY94xjVRaChc5g5EO?usp=drive_link](https://drive.google.com/drive/folders/1xhLDDFwJauVH5ijRY94xjVRaChc5g5EO?usp=drive_link)  
There you will find most of the files needed to begin topic modelling:  
* the `ARXIV_SNAPSHOT` as a `.json` file containing the metadata of all Arxiv preprints up including the 13th of November 2023
* the compressed `arxiv-metadata-oai-2023-11-13.json.gz` which is only needed if you wish to update your data at a later point in time
* a `.txt` file containint the resumptionToken necessary to do so
* the precalculated `embeddings.npy` obtained by encoding the preprocessed preprint abstracts with a sentence transformer
* the precalculated `reduced_embeddings.npy` obtained by using `cuML.manifold.umap` to reduce the embeddings down to only 5 dimensions
* the preprocessed `arxiv_df.feather` which contains the dataframe used for topic modelling after all preprocessing is done stored as `.feather` to drastically speed up loading of 2.3 million preprints to memory
Download the files and adjust the respective paths in `secrets.yaml`. It is recommended to leave all these files in a single folder since they will only be written to if the dataset is updated with `sciencenow.data.update_arxiv_snapshot.py`.


# https://github.com/MaartenGr/BERTopic_evaluation/blob/main/notebooks/Evaluation.ipynb

import pandas as pd
import numpy as np
# from bertopic import BERTopic
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import re
import string

ROOT = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/")
ROOT.exists()
ARXIV_PATH = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/arxivdata/arxiv-metadata-oai-snapshot.json")
ARXIV_PATH.exists()
ARTIFACTS = ROOT / Path("artifacts/antm")


dframe = pd.read_json(ARXIV_PATH, lines=True)

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df=dframe[["abstract","update_date"]].rename(columns={"abstract":"content","update_date":"time"})
df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
df['year'] = df['time'].dt.to_period('Y')
df=df.sort_values(by="year")

df['content'] = df['content'].str.replace(r'@\w+', '')
df['content'] = df['content'].str.replace('\n', ' ').replace('\r', '')
df['content'] = df['content'].apply(lambda x: remove_punct(x))

df = df.dropna()
df=df.sort_values(by="time")
df=df.reset_index(drop=True)
df=df.reset_index()


from antm import ANTM

window_size=3
overlap=1

#take a random sample for example
dt=df.sample(n = 10000)
dt=dt.sort_values("time")
dt=dt.rename(columns={"time":"date", "year":"time"})
dt=dt[["content", "time"]]
dt=dt.reset_index(drop=True)
dt=dt.reset_index()

model=ANTM(dt,overlap,window_size,mode="data2vec",num_words=10,path=str(ARTIFACTS))#
os.path.exists(str(ARTIFACTS) + "/results")

#learn the model and save it
model.fit(save=True)
"""
1: 789
2:   91011
3:      111213
4:          131415
5:              151617
6:                  181920
7:                      202122
8:                          2223
"""

diversity = model.get_periodwise_puw_diversity()
jaccard = model.get_periodwise_pairwise_jaccard_diversity()
coherence = model.get_periodwise_topic_coherence()
model.save_evolution_topics_plots()
model.plot_evolving_topics()
topics = model.random_evolution_topic()
model.plot_clusters_over_time()

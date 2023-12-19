import json
import pandas as pd
from pathlib import Path
from abc import ABC
from warnings import warn
import os
from matplotlib import pyplot as plt
import numpy as np

jsondir = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/tm_evaluation/Jan2021"
outpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/image.png"

class Visualizer(ABC):
    """
    Superclass to unify visualizing of topic model evaluation. 
    Expects the path to a root directory where .json files as produced by
    `sciencenow.models.train.ModelWrapper.train_and_evaluate`
    Params:
        jsondir: string that specifies the folder where evaluation results have been stored.
        outpath: string that specifies the absolute path where the resulting plot will be written to.
    """
    def __init__(
        self,
        jsondir,
        outpath,
        ):
        super().__init__()
        self.root = Path(jsondir)
        self.file_paths = self.get_paths()
        self.outpath = outpath
        self.fig = None

    def get_paths(self):
        if not self.root.exists():
            warn("Incorrect json directory specified.")
        else:
            file_list = [x for x in os.listdir(self.root) if x.endswith(".json")]
            absolute_paths = [self.root / x for x in file_list]
            return absolute_paths
     
    def plot(self):
        raise NotImplementedError

    def save_plot(self):
        outpath = Path(self.outpath)
        if not outpath.parent.exists():
            warn("Incorrect file path specified")
        elif self.fig is None:
            warn("Figure not found, call `plot()` first.")
        else:
            self.fig.savefig(self.outpath)

    def load_data(self):
        data = []
        for file in self.file_paths:
            with open(file) as f:
                data.append(json.load(f)[0])
        return data


class CoherenceVisualizer(Visualizer):
    """
    Plot Coherence scores of topic model against other parameter of choice.
    """
    def __init__(
        self,
        jsondir,
        outpath,
        ):
        super(CoherenceVisualizer, self).__init__(jsondir, outpath)
        self.data = self.load_data()

    def plot(self, param, xlabel, title):
        coherence = [x["Coherence"] for x in self.data]
        x_data = [x["Params"][f"{param}"] for x in self.data]
        d = {"Coherence": coherence, f"{param}": x_data}
        # needs to be sorted
        df = pd.DataFrame(data=d)
        df = df.sort_values(by=f"{param}")
        self.fig, self.ax = plt.subplots()
        self.ax.plot(df[f"{param}"], df["Coherence"])
        self.ax.set(
            xlabel=xlabel, 
            ylabel="Coherence (npmi)",
            title=f"NPMI vs {title} for {self.data[0]['Params']['startdate']}"
        )
        self.ax.grid()

class DiversityVisualizer(Visualizer):
    """
    Plot Coherence scores of topic model against other parameter of choice.
    """
    def __init__(
        self,
        jsondir,
        outpath,
        ):
        super(DiversityVisualizer, self).__init__(jsondir, outpath)
        self.data = self.load_data()

    def plot(self, param, xlabel, title):
        diversity = [x["Diversity"] for x in self.data]
        # need to average diversity since it is calculated separately for every time bin
        values = [list(x.values()) for x in diversity]
        div_averaged = []
        for entry in values:
            div_averaged.append(np.mean([x["diversity"] for x in entry]))
        x_data = [x["Params"][f"{param}"] for x in self.data]
        d = {"Diversity": div_averaged, f"{param}": x_data}
        # needs to be sorted
        df = pd.DataFrame(data=d)
        df = df.sort_values(by=f"{param}")
        self.fig, self.ax = plt.subplots()
        self.ax.plot(df[f"{param}"], df["Diversity"])
        self.ax.set(
            xlabel=xlabel, 
            ylabel="Diversity ()",
            title=f"Diversity vs {title} for {self.data[0]['Params']['startdate']}"
        )
        self.ax.grid()

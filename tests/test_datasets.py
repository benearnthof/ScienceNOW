from sciencenow.core.dataset import ArxivDataset

path = "C:\\Users\\Bene\\Desktop\\testfolder\\Experiments\\all-distilroberta-v1\\taxonomy.txt"

ds = ArxivDataset(path="path", pipeline=None)

ds.load_taxonomy(path=path)
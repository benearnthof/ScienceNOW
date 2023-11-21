"""Minimal Setup file to start training a topic model."""

from sciencenow.data.arxivprocessor import ArxivProcessor

processor = ArxivProcessor()

processor.load_snapshot()

startdate = "01 01 2020"
enddate = "31 12 2020"
target = "cs"
threshold = 100

subset = processor.filter_by_date_range(startdate=startdate, enddate=enddate) 
subset = processor.filter_by_taxonomy(subset=subset, target=target, threshold=threshold)

processor.bertopic_setup(subset=subset, recompute=True)

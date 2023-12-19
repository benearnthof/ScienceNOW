from sciencenow.postprocessing.eval import Visualizer, CoherenceVisualizer, DiversityVisualizer

# IF 
jsondir = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/tm_evaluation/Jan2021"
outpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/image.png"

# WHEN
viz = Visualizer(jsondir, outpath=outpath)

assert viz.root.exists()
assert viz.file_paths is not None
assert len(viz.file_paths) == 146
assert all([x.exists() for x in viz.file_paths])

# IF 
jsondir = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/tm_evaluation/Jan2021"
outpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/image.png"

viz = CoherenceVisualizer(jsondir, outpath=outpath)

viz.plot(param="cluster_size", xlabel="Minimum Cluster Size", title="HDBSCAN Cluster Size")
viz.save_plot()

viz = DiversityVisualizer(jsondir, outpath=outpath)
viz.plot(param="cluster_size", xlabel="Minimum Cluster Size", title="HDBSCAN Cluster Size")
viz.save_plot()

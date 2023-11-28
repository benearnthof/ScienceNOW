source /dss/dsshome1/lxc01/ru25jan4/miniconda3/bin/activate
conda activate /dss/dsshome1/lxc01/ru25jan4/miniconda3/envs/imagen
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo python3 version = `python3 --version`

cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/

max=200
for (( i=60; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/evaluate_bertopic.py -clust=$i
done

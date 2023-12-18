#!/bin/bash -l
#SBATCH --job-name=2021CLUST
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=benearnthof@hotmail.de
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=2021CLUST.out

source /dss/dsshome1/lxc01/ru25jan4/miniconda3/bin/activate
conda activate /dss/dsshome1/lxc01/ru25jan4/miniconda3/envs/imagen
# echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
# echo PATH = $PATH
# echo python3 version = `python3 --version`

# JAN
cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/
max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 01 2021" -edate="31 01 2021" -dir="Jan2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 02 2021" -edate="28 02 2021" -dir="Feb2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 03 2021" -edate="31 03 2021" -dir="Mar2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 04 2021" -edate="30 04 2021" -dir="Apr2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 05 2021" -edate="31 05 2021" -dir="May2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 06 2021" -edate="30 06 2021" -dir="Jun2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 07 2021" -edate="31 07 2021" -dir="Jul2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 08 2021" -edate="31 08 2021" -dir="Aug2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 09 2021" -edate="30 09 2021" -dir="Sep2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 10 2021" -edate="31 10 2021" -dir="Oct2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 11 2021" -edate="30 11 2021" -dir="Nov2021"
done

max=200
for (( i=1; i<=$max; ++i ))
do
    python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/eval_2021.py -clust=$i -sdate="01 12 2021" -edate="31 12 2021" -dir="Dec2021"
done

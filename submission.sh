#!/bin/bash
#PBS -A qum-543-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=1

source ./bin/activate
module purge -f
module load compilers/gcc/4.8.5 compilers/java/1.8 cuda/7.5 libs/cuDNN/5 apps/buildtools \
compilers/swig
cd aeproject
python train.py data/processed.h5 model.h5 --epochs 20
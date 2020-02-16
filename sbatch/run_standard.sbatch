#!/bin/bash

#SBATCH --partition awhite
#SBATCH --nodes 1
##SBATCH --exclusive
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time 0-24:00:00
#SBATCH --job-name nmrtf-std

ROOT_DIR=/scratch/awhite38/graphnmr
SCRATCH_DIR=$SCRATCH/tf/nmr

source ~/.bashrc
module load cudnn/9.0-7 zmq anaconda graphviz

source activate nmrtf
cd $SCRATCH_DIR
python $ROOT_DIR/scripts/train_standards.py $SCRATCH_DIR $ROOT_DIR/data

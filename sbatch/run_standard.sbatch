#!/bin/bash

#SBATCH --partition awhite
#SBATCH --nodes 1
##SBATCH --exclusive
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time 1-12:00:00
#SBATCH --job-name nmrs

ROOT_DIR=/scratch/awhite38/graphnmr
SCRATCH_DIR=$SCRATCH/tf/nmr

source ~/.bashrc
module load cudnn/9.0-7 zmq anaconda graphviz

source activate nmrtf
cd $SCRATCH_DIR
python $ROOT_DIR/scripts/train_standards.py $SCRATCH_DIR $ROOT_DIR/data ${JOB_NAME}

#!/bin/bash

#SBATCH --partition awhite
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time 0-24:00:00
#SBATCH --job-name nmrtf-data

ROOT_DIR=/scratch/awhite38/graphnmr
SCRATCH_DIR=$SCRATCH/

source ~/.bashrc
module load cudnn/9.0-7 zmq anaconda graphviz

source activate nmrtf
mkdir -p $SCRATCH_DIR/new_records/
cd $SCRATCH_DIR/new_records/
cp $ROOT_DIR/data/embeddings.pb .
python $ROOT_DIR/parse/protein_tfrecords.py $SCRATCH_DIR/nmr_data/proteins/

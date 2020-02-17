import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

DO_TRAIN = True

if len(sys.argv) == 3:
    SCRATCH = sys.argv[1]
    DATA_DIR = sys.argv[2]
else:    
    SCRATCH = os.curdir + os.path.sep
    DATA_DIR = 'records/'

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))

# read data from this file
filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord')]
atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

skips = [25000]

def train_model(name, hypers, stratify=False):
    print('Starting model', name)
    tf.reset_default_graph()
    hypers.LEARNING_RATE = tf.placeholder(tf.float32, shape=[])
    model = StructGCNModel(SCRATCH + name, embedding_dicts, hypers)
    model.build_from_datasets(create_datasets(filenames, skips),
        tf.constant([0], dtype=tf.int64), 20000,
        atom_number, neighbor_number)
    # add learning rate
    tf.summary.scalar('learning-rate', hypers.LEARNING_RATE)
    model.build_train()
    if DO_TRAIN:
        model.run_train({hypers.LEARNING_RATE: 1e-4})

    # Assess fit
    top1 = model.summarize_eval()
    print('Model top 1 error', top1)
    model.plot_examples(MAX_ATOM_NUMBER, top1, 25)


train_model('struct-model-11/tiny', GCNHypersTiny())
#train_model('struct-model-11/standard', GCNHypersStandard())
hypers = GCNHypersStandard()
hypers.EDGE_DISTANCE = False
train_model('struct-model-11/standard-nodist', hypers)
hypers.EDGE_NONBONDED = False
train_model('struct-model-11/standard-nononbond', hypers)


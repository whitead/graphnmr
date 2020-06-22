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
filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord')]
weighted_filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord')]
atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

skips = [25000]

def train_model(name, hypers, weighted=True):
    print('Starting model', name)
    tf.reset_default_graph()
    model = StructGCNModel(SCRATCH + name, embedding_dicts, hypers)
    model.build_from_datasets(create_datasets(weighted_filenames if weighted else filenames, skips),
        tf.constant([0], dtype=tf.int64), 20000,
        atom_number, neighbor_number)
    # add learning rate
    model.build_train()
    if DO_TRAIN:
        model.run_train()
    model.summarize_eval()

#train_model('struct-model-11/tiny', GCNHypersTiny())
train_model('struct-model-12/standard', GCNHypersStandard())
hypers = GCNHypersStandard()
#hypers.EDGE_DISTANCE = False
#train_model('struct-model-11/standard-nodist', hypers)
#hypers.EDGE_NONBONDED = False
#train_model('struct-model-11/standard-nononbond', hypers)


#hypers = GCNHypersStandard()
#hypers.BATCH_NORM = True
#hypers.GCN_BIAS = True
#hypers.GCN_RESIDUE = True
#hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
#train_model('struct-model-11/hyper-attempt-1', hypers, True)

hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = True
hypers.GCN_RESIDUE = True
#train_model('struct-model-11/hyper-attempt-2', hypers, True)

hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = False
hypers.GCN_RESIDUE = False
#train_model('struct-model-11/hyper-attempt-3', hypers, True)

hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = False
hypers.GCN_RESIDUE = False
#train_model('struct-model-11/hyper-attempt-4', hypers, False)
train_model('struct-model-11/hyper-attempt-4-weighted', hypers, True)


hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = True
hypers.GCN_RESIDUE = True
#train_model('struct-model-11/hyper-attempt-5', hypers, False)


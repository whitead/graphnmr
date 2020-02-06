import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

# THINGS TO MAKE BIG AGAIN:
# 1. Shulffle slize
# 2. Skip number
# 3. Batch size
# 4. Num batches

DO_TRAIN = True
DO_CHECKS = False

if len(sys.argv) == 3:
    SCRATCH = sys.argv[1]
    DATA_DIR = sys.argv[2]
else:    
    SCRATCH = os.curdir + os.path.sep
    DATA_DIR = 'records/'

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))

# read data from this file
filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord')]
#filenames = [DATA_DIR + f'train-structure-metabolite-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord']

if DO_CHECKS:
    validate_peaks(filenames[0], embedding_dicts)
    validate_embeddings(filenames[0], embedding_dicts)
    exit()
atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

hypers = GCNHypers()
hypers.NUM_EPOCHS = 1000
hypers.NUM_BATCHES = 256
hypers.BATCH_SIZE = 32
hypers.SAVE_PERIOD = 250
hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
hypers.STRATIFY = None
hypers.EDGE_DISTANCE = True
hypers.EDGE_NONBONDED = True
hypers.EDGE_LONG_BOND = True

skips = [25000]

def train_model(name, hypers):
    print('Starting model', name)
    tf.reset_default_graph()
    hypers.LEARNING_RATE = tf.placeholder(tf.float32, shape=[])
    model = StructGCNModel(SCRATCH + name, embedding_dicts, hypers)
    model.build_from_datasets(create_datasets(filenames, skips),
        tf.constant([0], dtype=tf.int64), 1000,
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

#train_model('struct-model-8/baseline', hypers)

hypers.GCN_RESIDUE = True
hypers.GCN_BIAS = True
hypers.BATCH_NORM = True
hypers.DROPOUT_RATE = 0.2
hypers.EDGE_FC_LAYERS = 3
hypers.FC_LAYERS = 4

hypers.EDGE_DISTANCE = False
hypers.EDGE_NONBONDED = False
hypers.EDGE_LONG_BOND = False
train_model('struct-model-8/bonds', hypers)

hypers.EDGE_NONBONDED = True
train_model('struct-model-8/nonbonds', hypers)


hypers.EDGE_LONG_BOND = True
#train_model('struct-model-8/kitchen-sink', hypers)

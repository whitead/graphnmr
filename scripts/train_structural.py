import tensorflow as tf
import pickle
import numpy as np
from model import *
import matplotlib.pyplot as plt
import os

# THINGS TO MAKE BIG AGAIN: 
# 1. Shulffle slize
# 2. Skip number
# 3. Batch size
# 4. Num batches

DO_TRAIN = True
DO_CHECKS = False

SCRATCH = os.curdir + os.path.sep
#SCRATCH = '/tmp/'
EMBEDDINGS_DIR = SCRATCH
DATA_DIR = 'records/'

embedding_dicts = load_embeddings('embeddings.pb')

# read data from this file
filenames = [DATA_DIR + f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord']
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
hypers.SAVE_PERIOD = 50
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


train_model('struct-model-5/distance', hypers)

#hypers.EDGE_DISTANCE = False
#train_model('struct-model-5/longbond', hypers)

#hypers.EDGE_NONBONDED = False
#train_model('struct-model-5/sbond', hypers)
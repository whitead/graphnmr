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

skips = [6000] # set to be about the same numebr of shifts as the test dataset

def train_model(name, hypers, weighted=True, learning_rates=None, restart=False):
    print('Starting model', name)
    tf.reset_default_graph()
    model = StructGCNModel(SCRATCH + name, embedding_dicts, hypers)
    model.build_from_datasets(create_datasets(weighted_filenames if weighted else filenames, skips),
        tf.constant([0], dtype=tf.int64), 20000,
        atom_number, neighbor_number)
    # add learning rate
    model.build_train()
    if DO_TRAIN:
        if learning_rates is None:
            model.run_train()
        else:
            model.hypers.LEARNING_RATE = learning_rates[0]
            model.run_train(restart=restart)
            for i, lr in enumerate(learning_rates[1:]):
                model.hypers.LEARNING_RATE = lr
                model.run_train(restart=True)
    model.summarize_eval()

#train_model('struct-model-11/tiny', GCNHypersTiny())
hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 8
hypers.NUM_EPOCHS = 10000
#train_model('struct-model-13/standard-unweighted', hypers, False)
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
#train_model('struct-model-13/hyper-attempt-2', hypers)

hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 8
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = False
hypers.GCN_RESIDUE = False
hypers.NUM_EPOCHS = 3000
#train_model('struct-model-13/hyper-attempt-3', hypers, learning_rates=[1e-4, 1e-6])

hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 8
hypers.ATOM_EMBEDDING_SIZE =  128
hypers.EDGE_EMBEDDING_SIZE = 16
hypers.EDGE_EMBEDDING_OUT = 8
#train_model('struct-model-13/hyper-attempt-4', hypers, True)


hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0.2
hypers.GCN_BIAS = True
hypers.RESIDUE = True
hypers.NON_LINEAR = True
hypers.FC_ACTIVATION = tf.keras.activations.softmax
hypers.GCN_ACTIVATION = tf.keras.activations.softmax
hypers.EDGE_EMBEDDING_SIZE = 16
hypers.EDGE_FC_LAYERS = 3
hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
hypers.SAVE_PERIOD = 25
hypers.NUM_EPOCHS = 3000
#train_model('struct-model-13/hyper-attempt-5', hypers, True, learning_rates=[1e-4, 1e-6])

hypers.NON_LINEAR = False
#train_model('struct-model-13/hyper-attempt-6', hypers, True, learning_rates=[1e-4, 1e-6])

hypers.NON_LINEAR = False
hypers.BATCH_SIZE = 16
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 1000
#train_model('struct-model-13/hyper-attempt-7', hypers, True, learning_rates=[1e-4, 1e-6, 1e-6, 1e-6, 1e-6])


hypers = GCNHypersStandard()
hypers.NON_LINEAR = True
hypers.BATCH_SIZE = 16
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 2000
#train_model('struct-model-13/hyper-attempt-8', hypers, True, learning_rates=[1e-4, 1e-6, 1e-6, 1e-6, 1e-6], restart=True)
#train_model('struct-model-13/hyper-attempt-8-unweighted', hypers, False, learning_rates=[1e-5, 1e-6, 1e-6, 1e-6, 1e-6], restart=True)

hypers = GCNHypersStandard()
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 5000
hypers.BATCH_NORM = True
#train_model('struct-model-13/hyper-attempt-9', hypers, True, learning_rates=[1e-4, 1e-5], restart=True)

hypers = GCNHypersStandard()
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 5000
hypers.BATCH_NORM = True
hypers.FC_ACTIVATION = tf.keras.activations.softmax
hypers.GCN_ACTIVATION = tf.keras.activations.softmax
#train_model('struct-model-13/hyper-attempt-10', hypers, True, learning_rates=[1e-4, 1e-5, 1e-6])


hypers = GCNHypersStandard()
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 2000
hypers.BATCH_NORM = True
hypers.ATOM_EMBEDDING_SIZE = 128
hypers.STACKS = 6
hypers.EDGE_FC_LAYERS = 4
hypers.EDGE_EMBEDDING_SIZE = 16
hypers.EDGE_EMBEDDING_OUT = 8
hypers.RESIDUE = False
hypers.BATCH_NORM = True
hypers.BATCH_SIZE = 8
#train_model('struct-model-13/hyper-attempt-11', hypers, True, learning_rates=[1e-6, 1e-6, 1e-6], restart=True)

hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 8
hypers.NUM_EPOCHS = 10000
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
train_model('struct-model-13/hyper-attempt-12', hypers, True, learning_rates=[1e-4, 1e-5, 1e-6], restart=True)

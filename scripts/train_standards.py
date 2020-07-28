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
with open(os.path.join(DATA_DIR,'peak_standards.pb'), 'rb') as f:
    peak_standards = pickle.load(f)

# read data from this file
filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord')]
metabolite = [os.path.join(DATA_DIR,f'train-structure-metabolite-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord')]
weighted_filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord'), os.path.join(DATA_DIR,f'train-structure-shift-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord')]
atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

skips = [6000] # set to be about the same numebr of shifts as the test dataset

def train_model(name, hypers, filenames, learning_rates=None, restart=False, skips=skips):
    print('Starting model', name)
    tf.reset_default_graph()
    model = StructGCNModel(SCRATCH + name, embedding_dicts, peak_standards, hypers)
    model.build_from_datasets(create_datasets(filenames, skips),
                              tf.constant([0], dtype=tf.int64), 20000,
                              atom_number, neighbor_number, predict_atom=None)
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

if sys.argv[3] == 'standard':
    hypers = GCNHypersStandard()
    train_model('struct-model-18/standard', hypers, weighted_filenames[:1], learning_rates=[1e-4, 1e-3, 1e-4])
    print('COMPLETED NORMAL DATASET')
    train_model('struct-model-18/standard', hypers, weighted_filenames[1:2], learning_rates=[1e-4, 1e-5], restart=True)
 
elif sys.argv[3] == 'nodist':
    hypers = GCNHypersStandard()
    hypers.EDGE_DISTANCE = False
    train_model('struct-model-18/nodist', hypers, weighted_filenames[:1], learning_rates=[1e-4, 1e-3, 1e-4])
    print('COMPLETED NORMAL DATASET')
    train_model('struct-model-18/nodist', hypers, weighted_filenames[1:2], learning_rates=[1e-4, 1e-5], restart=True)

elif sys.argv[3] == 'noneighs':
    hypers = GCNHypersStandard()
    hypers.EDGE_NONBONDED = False
    train_model('struct-model-18/noneighs', hypers, weighted_filenames[:1], learning_rates=[1e-4, 1e-3, 1e-4])
    print('COMPLETED NORMAL DATASET')
    train_model('struct-model-18/noneighs', hypers, weighted_filenames[1:2], learning_rates=[1e-4, 1e-5], restart=True)

elif sys.argv[3] == 'metabolite':
    hypers = GCNHypersStandard()
    hypers.BATCH_NORM = True
    hypers.DROPOUT = 0.0
    hypers.NUM_EPOCHS = 500
    train_model('struct-model-17/metabolite', hypers, metabolite[:1], learning_rates=[1e-1, 1e-3, 1e-4], restart=False, skips=[50])

else:
    raise InvalidArgumentError('Unkown job type')

import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

DO_TRAIN = True

if len(sys.argv) == 4:
    SCRATCH = sys.argv[1]
    DATA_DIR = sys.argv[2]
else:    
    print('Must pass 3 arguments')
    exit()

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))
with open(os.path.join(DATA_DIR,'peak_standards.pb'), 'rb') as f:
    peak_standards = pickle.load(f)

# read data from this file
filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord'), os.path.join(DATA_DIR,f'train-structure-shift-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord')]
metabolite = [os.path.join(DATA_DIR,f'train-structure-metabolite-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord')]
weighted_filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord'), os.path.join(DATA_DIR,f'train-structure-shift-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord')]
atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

skips = [0.2, 0.2] # set to be about the same numebr of shifts as the test dataset

def train_model(name, hypers, filenames, learning_rates=None, restart=False, skips=skips, atom=None):
    print('Starting model', name)
    tf.reset_default_graph()
    counts = [count_records(f) for f in filenames]
    for c,f in zip(counts, filenames):
        print(f'Found {c} in {f}')
    skips = [int(c * s) for c,s in zip(counts, skips)]
    hypers.NUM_BATCHES = counts[0] - skips[0]
    print(f'Adjusting NUM_BATCHES to be {hypers.NUM_BATCHES}. Will use {skips[0]} for validation data')
    model = StructGCNModel(SCRATCH + name, embedding_dicts, peak_standards, hypers)
    model.build_from_datasets(create_datasets(filenames, skips),
                              tf.constant([0], dtype=tf.int64), 20000,
                              atom_number, neighbor_number, predict_atom=atom)
    # add learning rate
    model.build_train()
    if DO_TRAIN:
        model.hypers.LEARNING_RATE = learning_rates[0]
        model.run_train(restart=restart, patience=5 if restart else 100) # no early stop on first run
        for i, lr in enumerate(learning_rates[1:]):
            model.hypers.LEARNING_RATE = lr
            model.run_train(restart=True)
    model.summarize_eval()

#train_model('struct-model-18/standard', hypers, weighted_filenames[:1], learning_rates=[1e-3, 1e-3], atom='H')

if sys.argv[3] == 'standard':
    hypers = GCNHypersStandard()
    train_model('struct-model-18/standard', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'standard-uw':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uw', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw', hypers, filenames[1:2], learning_rates=[1e-4, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'standard-uw2':
    hypers = GCNHypersStandard()
    hypers.EDGE_EMBEDDING_OUT = 16
    hypers.ATOM_EMBEDDING_SIZE =  64
    train_model('struct-model-18/standard-uw2', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'standard-uw3':
    hypers = GCNHypersStandard()
    hypers.EDGE_EMBEDDING_OUT = 16
    hypers.ATOM_EMBEDDING_SIZE =  128
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uw3', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw3', hypers, weighted_filenames[1:2], learning_rates=[1e-4, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'standard-uw4':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uw4', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw4', hypers, weighted_filenames[1:2], learning_rates=[1e-4, 1e-4, 1e-5], atom='H')


elif sys.argv[3] == 'standard-refdb':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-refdb', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-refdb', hypers, filenames[1:2], learning_rates=[1e-4, 1e-4, 1e-5], atom='H')


elif sys.argv[3] == 'standard-all':
    hypers = GCNHypersStandard()
    train_model('struct-model-18/standard-all', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5])

elif sys.argv[3] == 'soft':
    hypers = GCNHypersStandard()
    hypers.GCN_ACTIVATION = tf.keras.activations.softplus
    hypers.FC_ACTIVATION = tf.keras.activations.softplus
    train_model('struct-model-18/standard-soft', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')


elif sys.argv[3] == 'standard-md':
    hypers = GCNHypersMedium()
    train_model('struct-model-18/standard-md', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'standard-sm':
    hypers = GCNHypersSmall()
    train_model('struct-model-18/standard-sm', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'standard-tn':
    hypers = GCNHypersTiny()
    train_model('struct-model-18/standard-tn', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')


elif sys.argv[3] == 'nodist':
    hypers = GCNHypersStandard()
    hypers.EDGE_DISTANCE = False
    train_model('struct-model-18/nodist', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'noneighs':
    hypers = GCNHypersStandard()
    hypers.EDGE_NONBONDED = False
    train_model('struct-model-18/noneighs', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'linear':
    hypers = GCNHypersStandard()
    hypers.NON_LINEAR = False
    train_model('struct-model-18/linear', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'noresidue':
    hypers = GCNHypersStandard()
    hypers.RESIDUE = False
    train_model('struct-model-18/noresidue', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'dropout':
    hypers = GCNHypersStandard()
    hypers.DROPOUT_RATE = 0.2
    train_model('struct-model-18/dropout', hypers, weighted_filenames[1:2], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'metabolite':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/metabolite', hypers, metabolite[:1], learning_rates=[1e-1, 1e-3, 1e-4, 1e-4])

else:
    raise InvalidArgumentError('Unkown job type')

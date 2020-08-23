import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

DO_TRAIN = True
CURVE_POINTS = 10

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

def train_model(name, hypers, filenames, learning_rates=None, restart=False, skips=skips, atom=None, patience = 5, preload=None):
    print('Starting model', name)
    tf.reset_default_graph()
    counts = [count_records(f) for f in filenames]
    for c,f in zip(counts, filenames):
        print(f'Found {c} in {f}')
    skips = [int(c * s) for c,s in zip(counts, skips)]
    hypers.NUM_BATCHES = counts[0] - skips[0]
    print(f'Adjusting NUM_BATCHES to be {hypers.NUM_BATCHES}. Will use {skips} for validation data')
    model = StructGCNModel(SCRATCH + name, embedding_dicts, peak_standards, hypers)
    model.build_from_datasets(create_datasets(filenames, skips),
                              tf.constant([0], dtype=tf.int64), 20000,
                              atom_number, neighbor_number, predict_atom=atom)
    # add learning rate
    model.build_train()
    
    if DO_TRAIN:
        model.hypers.LEARNING_RATE = learning_rates[0]
        model.run_train(restart=restart, patience=patience if restart else 100, load_path=None if preload is None else SCRATCH + preload) # no early stop on first run
        for i, lr in enumerate(learning_rates[1:]):
            model.hypers.LEARNING_RATE = lr
            model.run_train(restart=True)
    model.summarize_eval()

#1e-3 -> really big
#1e-5 -> tuning
#1e-2 -> busted

if sys.argv[3] == 'standard':
    hypers = GCNHypersStandard()
    train_model('struct-model-18/standard', hypers, filenames[1:2], learning_rates=[1e-4, 1e-4, 1e-5], atom='H')

elif sys.argv[3] == 'standard-uw':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uw', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)


elif sys.argv[3] == 'standard-uw-rbf':
    hypers = GCNHypersStandard()
    hypers.EDGE_RBF = True
    hypers.EDGE_EMBEDDING_SIZE = 128
    #hypers.EDGE_EMBEDDING_OUT = 4
    #hypers.ATOM_EMBEDDING_SIZE = 256
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uw-rbf', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw-rbf', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'standard-w-rbf':
    hypers = GCNHypersStandard()
    hypers.EDGE_RBF = True
    hypers.EDGE_EMBEDDING_SIZE = 128
    #hypers.EDGE_EMBEDDING_OUT = 4
    #hypers.ATOM_EMBEDDING_SIZE = 256
    hypers.NUM_EPOCHS = 5
    #train_model('struct-model-18/standard-w-rbf', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5])
    hypers.NUM_EPOCHS = 50
    #train_model('struct-model-18/standard-w-rbf', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], restart=True)
    train_model('struct-model-18/standard-w-rbf', hypers, weighted_filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5])


elif sys.argv[3] == 'standard-uw-rbf-md':
    hypers = GCNHypersMedium()
    hypers.EDGE_RBF = True
    hypers.EDGE_EMBEDDING_SIZE = 128
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uw-rbf-md', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw-rbf-md', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)


elif sys.argv[3] == 'standard-uw-rbf-ef':
    hypers = GCNHypersStandard()
    hypers.EDGE_RBF = True
    hypers.EDGE_EMBEDDING_SIZE = 128
    hypers.EDGE_EMBEDDING_OUT = 8
    hypers.ATOM_EMBEDDING_SIZE = 64
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uw-rbf-ef', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw-rbf-ef', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'standard-uw-rbf-all':
    hypers = GCNHypersStandard()
    hypers.EDGE_RBF = True
    hypers.NUM_EPOCHS = 5
    hypers.EDGE_EMBEDDING_SIZE = 128
    train_model('struct-model-18/standard-uw-rbf-all', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5])
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uw-rbf-all', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], restart=True)





elif sys.argv[3] == 'standard-w':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-w', hypers, weighted_filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-w', hypers, weighted_filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)



elif sys.argv[3] == 'standard-uwc':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-uwc', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-uwc', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'standard-refdb-long':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-refdb-long', hypers, filenames[0:2], learning_rates=[1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-refdb-long', hypers, filenames[1:2], learning_rates=[1e-5], atom='H', restart=True)

elif sys.argv[3] == 'standard-all':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-all', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5])
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-all', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], restart=True)

elif sys.argv[3] == 'standard-all-extend':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-all-extend', hypers, weighted_filenames[1:2], learning_rates=[1e-5], restart=True, preload='struct-model-18/standard-all')

elif sys.argv[3] == 'standard-all2-extend':
    hypers = GCNHypersStandard()
    hypers.EMBEDDINGS_OUT = True
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-all2-extend', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], restart=True, preload='struct-model-18/standard-all2-refdb')

elif sys.argv[3] == 'standard-all2-metabolite':
    hypers = GCNHypersStandard()
    hypers.EMBEDDINGS_OUT = True
    hypers.NUM_EPOCHS = 50
    hypers.STRATIFY = True
    train_model('struct-model-18/standard-all2-metabolite', hypers, [filenames[1], metabolite[0]], learning_rates=[1e-4, 1e-5, 1e-5], restart=True, preload='struct-model-18/standard-all2-extend')



elif sys.argv[3] == 'standard-all2':
    hypers = GCNHypersStandard()
    hypers.EMBEDDINGS_OUT = True
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-all2', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5])
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-all2', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], restart=True)

elif sys.argv[3] == 'standard-all2-md':
    hypers = GCNHypersStandard()
    hypers.EMBEDDINGS_OUT = True
    hypers.ATOM_EMBEDDING_SIZE = 128
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-all2-md', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5])
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-all2-md', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], restart=True)



elif sys.argv[3] == 'standard-md':
    hypers = GCNHypersMedium()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-md', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-md', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'standard-sm':
    hypers = GCNHypersSmall()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-sm', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-sm', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'standard-tn':
    hypers = GCNHypersTiny()
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/standard-tn', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/standard-tn', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'nodist':
    hypers = GCNHypersStandard()
    hypers.EDGE_DISTANCE = False
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/nodist', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/nodist', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'noneighs':
    hypers = GCNHypersStandard()
    hypers.EDGE_NONBONDED = False
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/noneighs', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/noneighs', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)

elif sys.argv[3] == 'linear':
    hypers = GCNHypersStandard()
    hypers.NON_LINEAR = False
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/linear', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/linear', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)


elif sys.argv[3] == 'noresidue':
    hypers = GCNHypersStandard()
    hypers.RESIDUE = False
    hypers.NUM_EPOCHS = 5
    train_model('struct-model-18/noresidue', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], atom='H')
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/noresidue', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], atom='H', restart=True)


elif sys.argv[3] == 'metabolite':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 50
    train_model('struct-model-18/metabolite', hypers, metabolite[:1], learning_rates=[1e-1, 1e-3, 1e-4, 1e-4])

elif sys.argv[3] == 'curve-refdb':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 5
    points = np.exp(np.linspace(np.log(0.001), np.log(0.5),CURVE_POINTS))
    print('Generating training curve at', points)
    for i,p in enumerate(points):
        train_model(f'struct-model-18/curve-refdb-{i}', hypers, filenames[0:1], learning_rates=[1e-3, 1e-3, 1e-4, 1e-5], skips=[1 - p], atom='H')

elif sys.argv[3] == 'curve-shift':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 50
    points = np.exp(np.linspace(np.log(0.001), np.log(0.5),CURVE_POINTS))
    print('Generating training curve at', points)
    for i,p in enumerate(points):
        # load from standard uwc
        train_model(f'struct-model-18/curve-shift-{i}', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], skips=[1 - p], atom='H', restart=True, preload='struct-model-18/refdb-only')

elif sys.argv[3] == 'curve-shift-noload':
    hypers = GCNHypersStandard()
    hypers.NUM_EPOCHS = 50
    points = np.exp(np.linspace(np.log(0.001), np.log(0.5),CURVE_POINTS))
    print('Generating training curve at', points)
    for i,p in enumerate(points):
        # load from standard uwc
        train_model(f'struct-model-18/curve-shift-noload-{i}', hypers, filenames[1:2], learning_rates=[1e-4, 1e-5, 1e-5], skips=[1 - p], atom='H')
    
    


else:
    raise InvalidArgumentError('Unkown job type')

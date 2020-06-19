import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

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

def plot_model(name, hypers):
    print('Results for model ', name)
    tf.reset_default_graph()
    hypers.LEARNING_RATE = tf.placeholder(tf.float32, shape=[])
    model = StructGCNModel(SCRATCH + name, embedding_dicts, hypers)
    model.build_from_datasets(create_datasets(filenames, skips),
        tf.constant([0], dtype=tf.int64), 20000,
        atom_number, neighbor_number)
    model.build_train()
    # Assess fit
    top1, results = model.summarize_eval()
    with open(name.split('/')[-1] + '-results-summary.txt', 'w') as f:
        keys = ['title', 'corr-coeff', 'R^2', 'MAE', 'RMSD', 'N']
        fs = '{:<12} {:<8.4} {:<8.4} {:<8.4} {:<8.4} {:<8} '
        key_s = '{:<12} {:<8} {:<8} {:<8} {:<8} {:<8}'
        print(key_s.format(*keys))
        f.write(key_s.format(*keys) + '\n')
        for r in results:
            print(fs.format(*[r[k] for k in keys]))
            f.write(fs.format(*[r[k] for k in keys]) + '\n')


plot_model('struct-model-11/tiny', GCNHypersTiny())
#plot_model('struct-model-11/standard', GCNHypersStandard())
hypers10 = GCNHypersStandard()
hypers10.GCN_ACTIVATION = tf.keras.layers.LeakyReLU(0.1)
#plot_model('struct-model-10/baseline-3000', hypers10)
hypers = GCNHypersStandard()
hypers.EDGE_DISTANCE = False
#plot_model('struct-model-11/standard-nodist', hypers)
hypers.EDGE_NONBONDED = False
#plot_model('struct-model-11/standard-nononbond', hypers)
hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.GCN_BIAS = True
hypers.GCN_RESIDUE = True
hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
#plot_model('struct-model-11/hyper-attempt-1', hypers)

hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = True
hypers.GCN_RESIDUE = True
#plot_model('struct-model-11/hyper-attempt-2', hypers)

hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = False
hypers.GCN_RESIDUE = False
#plot_model('struct-model-11/hyper-attempt-3', hypers)

hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = False
hypers.GCN_RESIDUE = False
#plot_model('struct-model-11/hyper-attempt-4', hypers)
plot_model('struct-model-11/hyper-attempt-4-weighted', hypers)


hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = True
hypers.GCN_RESIDUE = True
#plot_model('struct-model-11/hyper-attempt-5', hypers)


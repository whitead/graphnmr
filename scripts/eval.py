import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

SCRATCH = sys.argv[1]
DATA_DIR = sys.argv[2]

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))
model_name = 'nmrstruct-model-8/kitchen-sink'

atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

hypers = GCNHypers()
hypers.NUM_EPOCHS = 1000
hypers.NUM_BATCHES = 256
hypers.BATCH_SIZE = 1
hypers.SAVE_PERIOD = 250
hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
hypers.STRATIFY = None
hypers.EDGE_DISTANCE = True
hypers.EDGE_NONBONDED = True
hypers.EDGE_LONG_BOND = True
hypers.GCN_RESIDUE = True
hypers.GCN_BIAS = True
hypers.BATCH_NORM = True
hypers.EDGE_FC_LAYERS = 3
hypers.EDGE_EMBEDDING_OUT = 8
hypers.ATOM_EMBEDDING_SIZE =  64



print('Preparing model', model_name)
tf.reset_default_graph()
model = StructGCNModel(SCRATCH + model_name, embedding_dicts, hypers)
model.build_from_dataset(sys.argv[3], atom_number=atom_number, neighbor_size=neighbor_number)
results = model.eval()
with open('evaluation.pb', 'wb') as f:
    pickle.dump(results, f)


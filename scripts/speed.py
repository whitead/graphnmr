import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

SCRATCH = sys.argv[1]
DATA_DIR = sys.argv[2]

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))

with open(os.path.join(DATA_DIR,'peak_standards.pb'), 'rb') as f:
    peak_standards = pickle.load(f)


atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

hypers = GCNHypersStandard()
model_name = 'nmrstruct-model-18/standard-all'

#hypers.EDGE_RBF = True
#hypers.EDGE_EMBEDDING_SIZE = 128
#hypers.EDGE_EMBEDDING_OUT = 8
#hypers.ATOM_EMBEDDING_SIZE = 128

#model_name = 'nmrstruct-model-18/standard-uw-rbf-all'



print('Preparing model', model_name)
tf.reset_default_graph()
model = StructGCNModel(SCRATCH + model_name, embedding_dicts, peak_standards, hypers)
model.build_from_dataset(sys.argv[3], gzip=True, atom_number=atom_number, neighbor_size=neighbor_number)
model.time_peaks()


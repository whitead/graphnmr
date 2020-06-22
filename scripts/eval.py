import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

SCRATCH = sys.argv[1]
DATA_DIR = sys.argv[2]

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))

atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 1
#hypers.BATCH_NORM = True
#hypers.GCN_BIAS = True
#hypers.GCN_RESIDUE = True
#hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
#model_name = 'nmrstruct-model-11/hyper-attempt-1'


#model_name = 'nmrstruct-model-11/standard'

model_name = 'nmrstruct-model-10/baseline-3000'




print('Preparing model', model_name)
tf.reset_default_graph()
model = StructGCNModel(SCRATCH + model_name, embedding_dicts, hypers)
model.build_from_dataset(sys.argv[3], gzip=False, atom_number=atom_number, neighbor_size=neighbor_number)
results = model.eval()
with open('evaluation.pb', 'wb') as f:
    pickle.dump(results, f)


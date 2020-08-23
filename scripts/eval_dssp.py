import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

SCRATCH = sys.argv[1]
DATA_DIR = sys.argv[2]
DSSP_DATA = sys.argv[3]

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))

with open(os.path.join(DATA_DIR,'peak_standards.pb'), 'rb') as f:
    peak_standards = pickle.load(f)


atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER



hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 1
hypers.BATCH_NORM = False
hypers.EMBEDDINGS_OUT = True
model_name = 'nmrstruct-model-18/standard-all2-md'

#hypers.EDGE_RBF = True
#hypers.EDGE_EMBEDDING_SIZE = 128
#hypers.EDGE_EMBEDDING_OUT = 8
hypers.ATOM_EMBEDDING_SIZE = 128

hypers = GCNHypersStandard()
model_name = 'nmrstruct-model-18/standard-all'

#hypers.EDGE_RBF = True
#hypers.EDGE_EMBEDDING_SIZE = 128
#hypers.EDGE_EMBEDDING_OUT = 8
#hypers.ATOM_EMBEDDING_SIZE = 128

#model_name = 'nmrstruct-model-18/standard-uw-rbf-all'


# get look-ups for residues
rinfo = dict()
with open(DSSP_DATA, 'r') as f:
    rinfo_table = np.loadtxt(sys.argv[3], skiprows=1, dtype='str')
    # convert to dict
    # key is record_index, value is dssp, class
    for i in range(rinfo_table.shape[0]):
        ri = tuple([int(v) for v in rinfo_table[i, 3:-1]])
        c = rinfo_table[i, 2]
        d = rinfo_table[i, -1]
        rinfo[ri] = (c, d)

# look-ups for atom and res
resdict = {v: k for k,v in embedding_dicts['class'].items()}
namedict = {v: k for k,v in embedding_dicts['name'].items()}



print('Preparing model', model_name)
tf.reset_default_graph()
model = StructGCNModel(SCRATCH + model_name, embedding_dicts, peak_standards, hypers)
model.build_from_dataset(sys.argv[4], gzip=True, atom_number=atom_number, neighbor_size=neighbor_number)
result = model.eval_small()

combos = dict()

for r in result:
    record_index = r['record_index']
    peaks = r['peaks']
    class_labels = r['class']
    names = r['names']
    peak_labels = r['peak_labels']
    mask = r['mask']
    indices = np.nonzero(mask)
    for b,i in zip(indices[0], indices[1]):
        ri = rinfo[tuple(record_index[b])]
        n = namedict[names[b,i]].split('-')[1]
        # only want HA
        if n != 'HA':
            continue
        c = resdict[class_labels[b,0]]
        assert c == ri[0], f'Class Mismatch {c} {ri} {record_index[b]}'
        k = (c, ri[1])
        if k not in combos:
            combos[k] = [], []
        combos[k][0].append(peak_labels[b, i])
        combos[k][1].append(peaks[b, i])


with open('dssp_results.pb', 'wb') as f:
    pickle.dump(combos, file=f)

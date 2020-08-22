from graphnmr import *
import os, sys
import numpy as np

if len(sys.argv) != 5:
    print('write_labels.py [records] [embeddings] [name_i] [name_j]')
    exit()

name_i, name_j = sys.argv[3], sys.argv[4]
embedding_dicts = load_embeddings(sys.argv[2])
result = find_pairs(sys.argv[1], embedding_dicts, name_i, name_j)
if len(result) == 0:
    print('none found')
np.savetxt(f'{name_i}-{name_j}.txt', result)

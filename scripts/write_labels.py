from graphnmr import *
import os, sys

if len(sys.argv) != 5:
    print('write_labels.py [embeddings] [record_info] [records] [out_file]')
    exit()

embedding_dicts = load_embeddings(sys.argv[1])
write_peak_labels(sys.argv[3], embedding_dicts, sys.argv[2], sys.argv[4])


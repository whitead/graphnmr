import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

DO_TRAIN = True # Turn off to plot

if len(sys.argv) == 3:
    SCRATCH = sys.argv[1]
    DATA_DIR = sys.argv[2]
else:    
    SCRATCH = os.curdir + os.path.sep
    DATA_DIR = 'records/'

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))

# read data from this file
filenames = [os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord')]
#filenames = [DATA_DIR + f'train-structure-metabolite-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord']


skips = [10000]

atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

hypers = GCNHypersStandard()
hypers.NUM_EPOCHS = 2000

def train_model(name, hypers):
    tf.reset_default_graph()
    hypers.LEARNING_RATE = tf.placeholder(tf.float32, shape=[])
    model = StructGCNModel(SCRATCH + name, embedding_dicts, hypers)
    model.build_from_datasets(create_datasets(filenames, skips),
        tf.constant([0], dtype=tf.int64), 1000,
        atom_number, neighbor_number)
    # add learning rate
    model.build_train()
    if DO_TRAIN:
        model.run_train({hypers.LEARNING_RATE: 1e-4})
    # Assess fit
    return model.eval_train()


embeds = [64, 128, 256, 512]
stacks = [2, 4, 8, 16]

fig, axs = plt.subplots(len(embeds), len(stacks), figsize=(14, 14), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
for i,e in enumerate(embeds):
    for j,s in enumerate(stacks):
        hypers.STACKS = s
        hypers.ATOM_EMBEDDING_SIZE = e
        p,l,c,n = train_model('struct-model-13/hypers-{}-{}'.format(e,s), hypers)
        p = np.array(p)
        l = np.array(l)
        ax = axs[i, j]
        if i == 0:
            ax.set_title(f'{s} Graph Convs', loc='center')
        if i == len(embeds) - 1:
            ax.set_xlabel('Measured Chemical Shift [ppm]')
        if j == 0:
            ax.set_ylabel(f'{e} Embeddings')
        if j == len(stacks) - 1:
            ax.twinx().set_ylabel('Measured Chemical Shift [ppm]')
        ax.scatter(l, p, marker='o', s=6, alpha=0.1, linewidth=0, color='#37d4e1')
        ax.plot([0,max(l)], [0,max(l)], '--', color='gray')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.text(1, 7, 'r = {:.3f}\nRMSE = {:.3f}'.format(np.corrcoef(p, l)[0,1], np.sqrt(np.mean((p - l)**2))), fontsize=12)
        print('completed', i, j)

for ax in fig.get_axes():
    ax.label_outer()
plt.savefig('comparison.png', dpi=300)
plt.show()


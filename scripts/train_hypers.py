import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os

DO_TRAIN = True

SCRATCH = os.curdir + os.path.sep
#SCRATCH = '/tmp/'
EMBEDDINGS_DIR = SCRATCH
DATA_DIR = 'records/'

embedding_dicts = load_embeddings('embeddings.pb')

# read data from this file
filenames = [DATA_DIR + f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord']
skips = [15000]

atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

hypers = GCNHypers()
hypers.NUM_EPOCHS = 1000
hypers.NUM_BATCHES = 256
hypers.BATCH_SIZE = 32
hypers.SAVE_PERIOD = 250
hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
hypers.STRATIFY = None
hypers.EDGE_DISTANCE = True
hypers.EDGE_NONBONDED = True
hypers.EDGE_LONG_BOND = True


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


embeds = [8, 16, 32, 64]
stacks = [2, 4, 8, 12]

fig, axs = plt.subplots(len(embeds), len(stacks), figsize=(14, 14), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
for i,e in enumerate(embeds):
    for j,s in enumerate(stacks):
        hypers.STACKS = s
        hypers.ATOM_EMBEDDING_SIZE = e
        p,l,c,n = train_model('struct-model-4/hypers-{}-{}'.format(e,s), hypers)
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
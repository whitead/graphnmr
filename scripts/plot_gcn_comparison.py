import tensorflow as tf
import pickle
from graphnmr import *
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = '/scratch/awhite38/tf/nmr/'

with open(SCRATCH + 'embeddings.pb', 'rb') as f:
    embedding_dicts = pickle.load(f)
# read data from this file
MAX_ATOM_NUMBER = 76
SCRATCH = '/scratch/awhite38/tf/nmr/'
filenames = [SCRATCH + 'data-76.tfrecord', SCRATCH + 'protein-data-76.tfrecord']
#filenames = ['records/data-76.tfrecord']

#This function defines the type of data we're reading
def _parse_function(proto):
    features = {'bond-data': tf.FixedLenFeature([MAX_ATOM_NUMBER, MAX_ATOM_NUMBER], tf.int64),
                'atom-data': tf.FixedLenFeature([MAX_ATOM_NUMBER], tf.int64),
                'peak-data': tf.FixedLenFeature([MAX_ATOM_NUMBER], tf.float32),
                'mask-data': tf.FixedLenFeature([MAX_ATOM_NUMBER], tf.float32)
               }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features['bond-data'], parsed_features['atom-data'], parsed_features['peak-data'], parsed_features['mask-data']

def test_gcn(embed, stacks):
    tf.reset_default_graph()
    #process it into types defined above and then we shuffle the data
    dataset = tf.data.TFRecordDataset(filenames).shuffle(10000).map(_parse_function)
    hypers = GCNHypersStandard()
    train_model = GCNModel(SCRATCH + 'comparison-model', embedding_dicts, hypers)
    atom_number = tf.placeholder(dtype=tf.int32, shape=[])
    train_model.hypers.NUM_EPOCHS = 51
    train_model.hypers.NUM_BATCHES = 64
    train_model.hypers.BATCH_SIZE = 128
    train_model.hypers.SAVE_PERIOD = 10
    train_model.hypers.LEARNING_RATE = 0.01
    train_model.hypers.STACKS = stacks
    train_model.hypers.ATOM_EMBEDDING_SIZE = embed
    train_model.hypers.LOSS_FUNCTION = tf.losses.mean_squared_error

    train_model.build_from_dataset(dataset, atom_number)
    train_model.build_train()

    train_model.run_train({atom_number: MAX_ATOM_NUMBER})
    train_model.hypers.NUM_EPOCHS = 51
    train_model.hypers.LEARNING_RATE = 1e-3
    train_model.hypers.LOSS_FUNCTION = tf.losses.huber_loss
    train_model.run_train({atom_number: MAX_ATOM_NUMBER}, restart=True)
    predict, labels = train_model.eval_train({atom_number: MAX_ATOM_NUMBER}, 10000)
    return np.array(predict), np.array(labels)

embeds = [2, 8, 12]
stacks = [2, 4, 6]

fig, axs = plt.subplots(len(embeds), len(stacks), figsize=(10, 10), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
for i,e in enumerate(embeds):
    for j,s in enumerate(stacks):
        p,l = test_gcn(e, s)
        ax = axs[i, j]
        if i == 0:
            ax.set_title(f'{s} Graph Convs', loc='center', fontsize=14)
        if j == 0:
            ax.set_ylabel(f'{e} Embeddings', fontsize=14)
        ax.scatter(p, l, marker='o', s=6, alpha=0.1, linewidth=0, color='C0')
        ax.plot([0,max(l)], [0,max(l)], '--', color='gray')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.text(1, 7, 'r = {:.3f}\nME = {:.3f}'.format(np.corrcoef(p, l)[0,1], np.mean(np.abs(p - l))), fontsize=12)
        print('completed', i, j)

for ax in fig.get_axes():
    ax.label_outer()
plt.savefig('comparison.png', dpi=300)
plt.show()

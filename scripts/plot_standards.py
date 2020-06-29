import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys

if len(sys.argv) >= 3:
    SCRATCH = sys.argv[1]
    DATA_DIR = sys.argv[2]
else:    
    SCRATCH = os.curdir + os.path.sep
    DATA_DIR = 'records/'

start_index = 0
if len(sys.argv) == 4:
    start_index = int(sys.argv[3])

embedding_dicts = load_embeddings(os.path.join(DATA_DIR,'embeddings.pb'))
model_dir = 'struct-model-13'

# read data from this file
test_file = os.path.join(DATA_DIR,f'test/test-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}.tfrecord')
train_file = os.path.join(DATA_DIR,f'train-structure-protein-data-{MAX_ATOM_NUMBER}-{NEIGHBOR_NUMBER}-weighted.tfrecord')

atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

skips = [10000]


def plot_model(name, hypers, data='test', progressive=False):
    print('Results for model ', name)
    plot_dir = ''
    tf.reset_default_graph()
    model = StructGCNModel(SCRATCH + name, embedding_dicts, hypers)    
    test = False
    if data == 'test':
        test =True
        print('Evaluating test data')
        model.build_from_dataset(test_file, True, atom_number=atom_number, neighbor_size=neighbor_number)
        plot_dir = 'test'
    else:
        train = False
        if data == 'train':
            train = True
            plot_dir = 'train'
            print('Evaluating train data')
        else:
            plot_dir = ''
            print('Evaluating validation data')
        model.build_from_datasets(create_datasets([train_file], skips, train),
        tf.constant([0], dtype=tf.int64), 20000,
        atom_number, neighbor_number)
        model.build_train()
    # Assess fit
    top1, results = model.summarize_eval(test_data=test, plot_dir_name='plots-' + plot_dir)
    with open(name.split('/')[-1] + '-{}-results-summary.txt'.format(plot_dir), 'w') as f:
        keys = ['title', 'corr-coeff', 'R^2', 'MAE', 'RMSD', 'N']
        fs = '{:<12} {:<8.4} {:<8.4} {:<8.4} {:<8.4} {:<8} '
        key_s = '{:<12} {:<8} {:<8} {:<8} {:<8} {:<8}'
        print(key_s.format(*keys))
        f.write(key_s.format(*keys) + '\n')
        for r in results:
            print(fs.format(*[r[k] for k in keys]))
            f.write(fs.format(*[r[k] for k in keys]) + '\n')
    # progressive plots
    if progressive:
     for i in range(start_index, 1000):
         try:
             model.summarize_eval(test_data=test, checkpoint_index=i, classes=False, plot_dir_name='plots-' + plot_dir)
         except StopIteration as e:
             break
        


#plot_model(model_dir + '/tiny', GCNHypersTiny())
hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 8
#plot_model(model_dir + '/standard', hypers, progressive=False)
#plot_model(model_dir + '/standard-unweighted', hypers, progressive=False)
#plot_model(model_dir + '/standard', hypers, False, progressive=False)

hypers = GCNHypersStandard()
hypers.EDGE_DISTANCE = False
#plot_model(model_dir + '/standard-nodist', hypers)
hypers.EDGE_NONBONDED = False
#plot_model(model_dir + '/standard-nononbond', hypers)
hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.GCN_BIAS = True
hypers.GCN_RESIDUE = True
hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
#plot_model(model_dir + '/hyper-attempt-1', hypers)

hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = True
hypers.GCN_RESIDUE = True
#plot_model(model_dir + '/hyper-attempt-2', hypers, progressive=False)


hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 8
hypers.BATCH_NORM = True
hypers.DROPOUT = 0
hypers.GCN_BIAS = False
hypers.GCN_RESIDUE = False
#plot_model(model_dir + '/hyper-attempt-3', hypers)

hypers = GCNHypersStandard()
hypers.BATCH_SIZE = 8
hypers.ATOM_EMBEDDING_SIZE =  128
hypers.EDGE_EMBEDDING_SIZE = 16
hypers.EDGE_EMBEDDING_OUT = 8
#plot_model(model_dir + '/hyper-attempt-4', hypers)





hypers = GCNHypersStandard()
hypers.BATCH_NORM = True
hypers.DROPOUT = 0.2
hypers.GCN_BIAS = True
hypers.RESIDUE = True
hypers.NON_LINEAR = True
hypers.FC_ACTIVATION = tf.keras.activations.softmax
hypers.GCN_ACTIVATION = tf.keras.activations.softmax
hypers.EDGE_EMBEDDING_SIZE = 16
hypers.EDGE_FC_LAYERS = 3
hypers.LOSS_FUNCTION = tf.losses.mean_squared_error
hypers.SAVE_PERIOD = 25
hypers.NUM_EPOCHS = 3000
#plot_model(model_dir + '/hyper-attempt-5', hypers, progressive=False)

hypers.NON_LINEAR = False
#plot_model(model_dir + '/hyper-attempt-6', hypers, progressive=False)


hypers = GCNHypersStandard()
hypers.NON_LINEAR = True
hypers.BATCH_SIZE = 16
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 1000
#plot_model(model_dir + '/hyper-attempt-8', hypers, progressive=False)
plot_model(model_dir + '/hyper-attempt-8', hypers, progressive=False, data='train')
plot_model(model_dir + '/hyper-attempt-8-unweighted', hypers, progressive=False, data='train')
#plot_model(model_dir + '/hyper-attempt-8', hypers, progressive=False)
#plot_model(model_dir + '/hyper-attempt-8-unweighted', hypers, progressive=False)


hypers = GCNHypersStandard()
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 5000
hypers.BATCH_NORM = True
plot_model('struct-model-13/hyper-attempt-9', hypers)
plot_model('struct-model-13/hyper-attempt-9', hypers, data='train')


hypers = GCNHypersStandard()
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 5000
hypers.BATCH_NORM = True
hypers.FC_ACTIVATION = tf.keras.activations.softmax
hypers.GCN_ACTIVATION = tf.keras.activations.softmax

plot_model('struct-model-13/hyper-attempt-10', hypers)
plot_model('struct-model-13/hyper-attempt-10', hypers, data='train')


hypers = GCNHypersStandard()
hypers.SAVE_PERIOD = 50
hypers.NUM_EPOCHS = 2000
hypers.BATCH_NORM = True
hypers.ATOM_EMBEDDING_SIZE = 128
hypers.STACKS = 6
hypers.EDGE_FC_LAYERS = 4
hypers.EDGE_EMBEDDING_SIZE = 16
hypers.EDGE_EMBEDDING_OUT = 8
hypers.EDGE_EMBEDDING_OUT = 8
hypers.RESIDUE = False
hypers.BATCH_NORM = True
hypers.BATCH_SIZE = 8
plot_model('struct-model-13/hyper-attempt-11', hypers)
plot_model('struct-model-13/hyper-attempt-11', hypers, data='train')

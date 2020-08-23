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
with open(os.path.join(DATA_DIR,'peak_standards.pb'), 'rb') as f:
    peak_standards = pickle.load(f)

atom_number = MAX_ATOM_NUMBER
neighbor_number = NEIGHBOR_NUMBER

# read data from this file
test_file = os.path.join(DATA_DIR,f'test/test-structure-shift-data-{atom_number}-{neighbor_number}.tfrecord')
train_file = os.path.join(DATA_DIR,f'train-structure-protein-data-{atom_number}-{neighbor_number}.tfrecord')
#valid_file = os.path.join(DATA_DIR,f'train-structure-shift-data-{atom_number}-{neighbor_number}-weighted.tfrecord')
valid_file = train_file



skips = [6000]


def plot_model(name, hypers, data='test', progressive=False, atom=None):
    print('Results for model ', SCRATCH + name)
    plot_dir = ''
    tf.reset_default_graph()
    model = StructGCNModel(SCRATCH + name, embedding_dicts, peak_standards, hypers)
    test = False
    if data == 'test':
        test =True
        print('Evaluating test data')
        model.build_from_dataset(test_file, True, 
                                 atom_number=atom_number, neighbor_size=neighbor_number,
                                 predict_atom=atom)
        plot_dir = 'test'
    else:
        train = False
        if data == 'train':
            f = train_file
            train = True
            plot_dir = 'train'
            print('Evaluating train data')
        else:
            f = valid_file
            plot_dir = 'validation'
            print('Evaluating validation data')
        model.build_from_datasets(create_datasets([f], skips, train),
        tf.constant([0], dtype=tf.int64), 20000,
        atom_number, neighbor_number)
        model.build_train()
    tpn = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('TRAINABLE PARAMETERS:', tpn)
    # Assess fit
    try:
        top1, results = model.summarize_eval(test_data=test, plot_dir_name='plots-' + plot_dir)
    except AttributeError:
        print('Skipping', name, 'probably because no checkpoints yet')
        return
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
        

model_dir = 'struct-model-18'

hypers = GCNHypersStandard()
hypers.EMBEDDINGS_OUT = True
plot_model(model_dir + '/standard-all2', hypers)

hypers = GCNHypersStandard()
hypers.EMBEDDINGS_OUT = True
plot_model(model_dir + '/standard-all2-refdb', hypers)


hypers = GCNHypersStandard()
hypers.EMBEDDINGS_OUT = True
plot_model(model_dir + '/standard-all2-extend', hypers)

exit()

hypers = GCNHypersStandard()
hypers.EMBEDDINGS_OUT = True
plot_model(model_dir + '/standard-all2-metabolite', hypers)



hypers.EMBEDDINGS_OUT = True
hypers.ATOM_EMBEDDING_SIZE = 128
plot_model(model_dir + '/standard-all2-md', hypers)




hypers = GCNHypersStandard()
hypers.EDGE_RBF = True
hypers.EDGE_EMBEDDING_SIZE = 128
#hypers.EDGE_EMBEDDING_OUT = 8
#hypers.ATOM_EMBEDDING_SIZE = 128
plot_model(model_dir + '/standard-uw-rbf', hypers, atom='H')
plot_model(model_dir + '/standard-w-rbf', hypers)



hypers = GCNHypersStandard()
hypers.EDGE_RBF = True
hypers.EDGE_EMBEDDING_SIZE = 128
hypers.EDGE_EMBEDDING_OUT = 8
hypers.ATOM_EMBEDDING_SIZE = 64
plot_model(model_dir + '/standard-uw-rbf-ef', hypers, atom='H')


hypers = GCNHypersStandard()
hypers.EDGE_RBF = True
hypers.EDGE_EMBEDDING_SIZE = 128
plot_model(model_dir + '/standard-uw-rbf-all', hypers)


exit()


hypers = GCNHypersStandard()
plot_model(model_dir + '/standard-all', hypers)

hypers = GCNHypersStandard()
plot_model(model_dir + '/standard-w', hypers, atom='H')

hypers = GCNHypersStandard()
plot_model(model_dir + '/standard-uw', hypers, atom='H')


hypers = GCNHypersStandard()
plot_model(model_dir + '/standard', hypers, atom='H')

hypers = GCNHypersStandard()
plot_model(model_dir + '/standard-refdb-long', hypers, atom='H')

hypers = GCNHypersStandard()
plot_model(model_dir + '/refdb-only', hypers, atom='H')

hypers = GCNHypersStandard()
plot_model(model_dir + '/standard-uwc', hypers, atom='H')



hypers = GCNHypersStandard()
hypers.NON_LINEAR = False
plot_model(model_dir + '/linear', hypers, atom='H')

hypers = GCNHypersStandard()
hypers.EDGE_DISTANCE = False
plot_model(model_dir + '/nodist', hypers , atom = 'H')

hypers = GCNHypersStandard()
hypers.EDGE_NONBONDED = False
plot_model(model_dir + '/noneighs', hypers, atom='H')

hypers = GCNHypersSmall()
plot_model(model_dir + '/standard-sm', hypers, atom='H')

hypers = GCNHypersMedium()
plot_model(model_dir + '/standard-md', hypers, atom='H')

hypers = GCNHypersTiny()
#plot_model(model_dir + '/standard-tn', hypers, atom='H')

hypers = GCNHypersStandard()
hypers.RESIDUE = False
#plot_model(model_dir + '/noresidue', hypers, atom='H')


hypers = GCNHypersStandard()
for i in range(20):
    plot_model(model_dir + f'/curve-shift-{i}', hypers, atom='H')

for i in range(20):
    plot_model(model_dir + f'/curve-refdb-{i}', hypers, atom='H')

for i in range(20):
    plot_model(model_dir + f'/curve-shift-noload-{i}', hypers, atom='H')

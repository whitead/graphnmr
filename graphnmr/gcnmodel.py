import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tqdm
from .data import data_parse
from .losses import *
from .layers import *


def safe_div(numerator, denominator, name='graphbuild-safe-div'):
    '''Divides two values, returning 0 if the denominator is <= 0.
    Args:
        numerator: A real `Tensor`.
        denominator: A real `Tensor`, with dtype matching `numerator`.
        name: Name for the returned op.
    Returns:
        0 if `denominator` <= 0, else `numerator` / `denominator`
    Taken from tensorflow/contrib/metrics/python/ops/metric_ops.py
    '''
    with tf.name_scope('graphbuilder-safe-div'):
        op = tf.where(
            tf.greater(denominator, 0),
            tf.truediv(numerator, denominator),
            tf.zeros_like(denominator),
            name=name)

    return op

class GCNHypers:
    def __init__(self):
        self.ATOM_EMBEDDING_SIZE =  16 #Size of space onto which we project elements
        self.EDGE_EMBEDDING_SIZE = 4 #size of space onto which we poject bonds (single, double, etc.)
        self.EDGE_RBF = False #size of space onto which we poject bonds (single, double, etc.)
        self.EDGE_EMBEDDING_OUT = 4 # what size edges are used in final model
        self.EMBEDDINGS_OUT = False 
        self.BATCH_SIZE = 32 #Amount of data we process at a time, in units of molecules (not atoms!)
        self.STACKS = 4 #Number of layers in graph convolution
        self.FC_LAYERS = 3
        self.EDGE_FC_LAYERS = 2
        self.NUM_EPOCHS = 100 #Number of times we do num_batches (how often we stop and save model basically)
        self.NUM_BATCHES = 500 #Number of batches between saves
        self.RESIDUE = True
        self.GCN_BIAS = False
        self.BATCH_NORM = False
        self.NON_LINEAR = True
        self.GCN_ACTIVATION = tf.keras.activations.relu # This change needs to be validated as irrelevant tf.keras.layers.LeakyReLU(0.1)
        self.FC_ACTIVATION = tf.keras.activations.relu
        #self.LOSS_FUNCTION = tf.losses.huber_loss
        self.LOSS_FUNCTION = corr_loss
        #self.LOSS_FUNCTION = tf.losses.absolute_difference
        self.LEARNING_RATE = 1e-4
        self.DROPOUT_RATE = 0.0
        self.GCN_DROPOUT = True
        self.SAVE_PERIOD = 1
        self.STRATIFY = False
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.REGULARIZER = None #tf.keras.regularizers.L1L2(l2=0.0)
        self.PEAK_CLIP = 500 # clip peaks at this. Some garbage data always gets through it seems.

class GCNModel:
    def __init__(self, model_path, embedding_dicts, peak_standards, hypers = None):
        if hypers is None:
            hypers = GCNHypers()
        self.hypers = hypers
        self.model_path = model_path
        self.embedding_dicts = embedding_dicts
        self.built = False
        self.using_dataset = False
        self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[])
        self.peak_standards = peak_standards
        self.global_steps = 0

    def build_from_datasets(self, datasets, dataset_choices=None, shuffle=10000, *args, **kw_args):

        print('Building from datasets {}. Will shuffle with buffer {}...'.format(datasets, shuffle), end='')
        # datasets should be tuples with train, test        
        train_dataset = datasets[0][0]
        test_dataset = datasets[0][1]
        
        # add additional if  present
        for d in datasets[1:]:
            print('Combing multiple datasets')
            train_dataset = train_dataset.concatenate(d[0])
            test_dataset = test_dataset.concatenate(d[1])


        # make a train and test dataset, where test is size TEST_N and train is size BATCH_SIZE repeated indefinitely
        # This removes TEST_N data points for validation. Then we say repeat test data as many times as wanted
        if self.hypers.STRATIFY:
            target_dist = [1 / len(self.embedding_dicts['class']) for _ in self.embedding_dicts['class']]
            print('Will stratify dataset to', target_dist)
            train_dataset = train_dataset.apply(tf.data.experimental.rejection_resample(lambda *x: tf.reshape(x[5], []), target_dist))
            # drop extra class (added by rejection sample
            train_dataset = train_dataset.map(lambda _, x: x)

        test_dataset = test_dataset.shuffle(shuffle // 4).batch(self.hypers.BATCH_SIZE)
        train_dataset = train_dataset.shuffle(shuffle).batch(self.hypers.BATCH_SIZE)

        # Now we make an iterator which allows us to view different batches of data
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.train_init_op = iterator.make_initializer(train_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)

        # assume this order
        (bond_inputs, atom_inputs, peak_inputs, mask_inputs, name_inputs, class_input, record_index) = iterator.get_next()
        self.raw_mask = mask_inputs # we will filter by atom type later
        self.peak_labels = tf.clip_by_value(tf.where(tf.is_nan(peak_inputs), tf.zeros_like(peak_inputs), peak_inputs), 0, self.hypers.PEAK_CLIP)
        self.class_label = class_input
        self.names = name_inputs
        self.record_index = record_index
        self.atoms = atom_inputs

        self.using_dataset = True

        print('done')

        self.build(atom_inputs, bond_inputs, *args, **kw_args)

    def build_from_dataset(self, filename, gzip, *args, **kw_args):

        tf_dataset = tf.data.TFRecordDataset([filename], compression_type='GZIP' if gzip else None).map(data_parse).batch(self.hypers.BATCH_SIZE)

        # Now we make an iterator which allows us to view different batches of data
        iterator = tf.data.Iterator.from_structure(tf_dataset.output_types, tf_dataset.output_shapes)
        self.train_init_op = None
        self.test_init_op = iterator.make_initializer(tf_dataset)

        # assume this order
        (bond_inputs, atom_inputs, peak_inputs, mask_inputs, name_inputs, class_input, record_index) = iterator.get_next()
        self.raw_mask = mask_inputs # we will filter by atom type later
        self.peak_labels = tf.clip_by_value(tf.where(tf.is_nan(peak_inputs), tf.zeros_like(peak_inputs), peak_inputs), 0, self.hypers.PEAK_CLIP)
        self.class_label = class_input
        self.names = name_inputs
        self.record_index = record_index
        self.atoms = atom_inputs
        self.using_dataset = True
        self.build(atom_inputs, bond_inputs, *args, **kw_args)


    def _feed_dict(self, fd, training):
        return {**fd, self.dropout_rate:self.hypers.DROPOUT_RATE if training else 0, self.training: training}


    def load(self, sess, i=-1, load_path=None):
        saver = tf.train.Saver()
        if load_path is None:
            load_path = self.model_path
        if i < 0:
            checkpoint = tf.train.latest_checkpoint(load_path)
        else:
            checkpoints = tf.train.get_checkpoint_state(load_path).all_model_checkpoint_paths
            if i >= len(checkpoints):
                raise StopIteration
            checkpoint = checkpoints[i]
        print(f'Reading checkpoint from {load_path}')
        step = int(checkpoint.split('/')[-1].split('.')[-1].split('-')[-1])
        print('Restoring checkpoint {} @ step {}'.format(checkpoint, step))
        saver.restore(sess, checkpoint)
        self.global_step = step

    def build_train(self, histogram_peak_names=False):
        if not self.built:
            raise ValueError('Must build before build_train')
        # histogram classes so we can check stratification of data
        class_number = len(self.embedding_dicts['class'])
        class_counts = tf.get_variable('classes', initializer=tf.zeros([class_number]))
        obs_classes = tf.reshape(tf.one_hot(self.class_label, depth=class_number), [-1, class_number])
        update_op = class_counts.assign_add(tf.reduce_sum(obs_classes, axis=0))
        self.reset_counts = class_counts.assign(tf.zeros_like(class_counts))
        # get things to regularize outside of keras
        # hide it when not training in case we're assessing test loss
        if self.hypers.REGULARIZER:
            reg_penalty = tf.cast(self.training, tf.float32) * tf.reduce_sum([self.hypers.REGULARIZER(v) for v in self.weights])
        else:
            reg_penalty = 0.0
        #reg_penalty = tf.cast(self.training, tf.float32) * tf.reduce_mean(self.hypers.REGULARIZER(self.nlist_grad))

        with tf.control_dependencies([update_op]):
            # we work in standardize labels for training
            # those are output by raw_peaks
            trans_labels = safe_div(self.peak_labels  - tf.nn.embedding_lookup(self.peak_avg, self.features), tf.nn.embedding_lookup(self.peak_std, self.features))
            self.loss = self.hypers.LOSS_FUNCTION(labels=trans_labels, predictions=self.raw_peaks, weights=self.mask) + reg_penalty
            optimizer = tf.train.AdamOptimizer(self.hypers.LEARNING_RATE)
            #gvs =  optimizer.compute_gradients(self.loss)
            #gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
            #self.train_step = optimizer.apply_gradients(gvs)
            self.train_step = optimizer.minimize(self.loss)
            self.class_counts = tf.identity(class_counts)
        tf.summary.scalar('loss', self.loss)
        self.corr = corr_coeff(self.peak_labels, self.peaks, self.mask)**2
        tf.summary.scalar('r2', self.corr)
        self.masked_peaks = tf.boolean_mask(self.peaks, self.bool_mask)
        self.masked_labels = tf.boolean_mask(self.peak_labels, self.bool_mask)
        tf.summary.histogram('peak-error', self.masked_peaks - self.masked_labels)
        tf.summary.histogram('classes', self.class_label)

        # write out embeddings metadata
        embedding_dir = '/logdir/test/embeddings/'
        for k,v in self.embedding_dicts.items():
            os.makedirs(self.model_path + embedding_dir, exist_ok=True)
            with open(self.model_path + embedding_dir + k + '.tsv', 'w') as f:
                # need to sort them
                embedding = list(v.keys())
                embedding.sort(key = lambda e: v[e])
                for e in embedding:
                    f.write('{}\n'.format(e))

        # setup embedding projector
        self.embedding_projector_config = projector.ProjectorConfig()

        # Add embedding to the projector
        pe = self.embedding_projector_config.embeddings.add()
        pe.tensor_name = self.atom_embeddings.name
        pe.metadata_path = 'embeddings/atom.tsv'

        # Look at histograms
        if histogram_peak_names:
            tf.summary.histogram('atom-names', self.names)
            for k,v in self.embedding_dicts['name'].items():
                print('Adding histogram for ', k)
                name_mask = tf.math.equal(tf.constant(v, dtype=tf.int64), self.names)
                net_mask = tf.math.logical_and(name_mask, self.bool_mask)
                tf.summary.scalar(k + '-count', tf.reduce_sum(tf.cast(net_mask, tf.int32)))
                tf.summary.histogram(k + '-error', tf.boolean_mask(self.peaks - self.peak_labels, net_mask))



    def run_train(self, feed_dict={}, restart=False, patience=3, trailing_avg=5, load_path = None):
        if not self.built:
            raise ValueError('Must build first')
        test_losses = []
        train_losses = []
        cur_patience = patience
        self.test_counts = [0 for c in self.embedding_dicts['class'].keys()]
        self.train_counts = self.test_counts[:]

        saver = tf.train.Saver(max_to_keep=self.hypers.NUM_EPOCHS)
        config = tf.ConfigProto()
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.model_path + '/logdir/train', sess.graph)
            test_writer = tf.summary.FileWriter(self.model_path + '/logdir/test')
            # create embedding visualizer
            projector.visualize_embeddings(test_writer  , self.embedding_projector_config)
            init = tf.global_variables_initializer()
            print('Initializing variables...', end='')
            sess.run(init)
            print('done')
            if restart:
                self.load(sess, load_path=load_path)

            sess.run([self.train_init_op, self.reset_counts])
            single_iter_count = 0
            for epoch in range(self.hypers.NUM_EPOCHS):
                print('', flush=True)
                try:
                    for batch in range(self.hypers.NUM_BATCHES):
                        _, loss, counts = sess.run([self.train_step, self.loss, self.class_counts], feed_dict=self._feed_dict(feed_dict, True))
                        # summed already in code.
                        self.train_counts = counts
                        single_iter_count += self.hypers.BATCH_SIZE
                        self.global_steps += 1
                        train_losses.append(loss)
                    print('epoch {} loss = {}'.format(epoch, loss))
                    # compute training summary
                    train_summary = sess.run(merged, feed_dict=self._feed_dict(feed_dict, True))
                    train_writer.add_summary(train_summary, self.global_steps)
                except tf.errors.OutOfRangeError as e:
                    sess.run([self.train_init_op, self.reset_counts])
                    print('Completed due to end of training data {}'.format(single_iter_count))
                    single_iter_count = 0
                if epoch % self.hypers.SAVE_PERIOD == 0 or epoch == self.hypers.NUM_EPOCHS - 1:
                    saver.save(sess, self.model_path + '/model.ckpt', global_step=self.global_steps)
                    # now run test
                    sess.run([self.test_init_op, self.reset_counts])
                    loss, counts, test_summary = sess.run([self.loss, self.class_counts, merged], feed_dict=self._feed_dict(feed_dict, False))
                    self.test_counts += counts
                    test_losses.append(loss)
                    # check for early stop
                    if len(test_losses) > 2 * trailing_avg:
                        cur_avg = sum(test_losses[-trailing_avg:])
                        prev_avg = sum(test_losses[-2 * trailing_avg:-trailing_avg])
                        print(f'current trailing loss vs previous: {cur_avg}, {prev_avg}')
                        if cur_avg > prev_avg:
                            cur_patience -= 1
                            if cur_patience < 0:
                                # early stop
                                print('No more patience, stopping')
                                break
                            else:
                                print(f'Being patient {cur_patience} more times')
                        else:
                            print(f'Improved, Patience was {cur_patience}')
                            cur_patience = patience
                    # only write summary on save periods.
                    test_writer.add_summary(test_summary, self.global_steps)
                    # save another one for the embedding projector (but overwrite)
                    saver.save(sess, self.model_path + '/logdir/test/model.ckpt', 0)
                    print(epoch, 'test:' ,test_losses[-1], 'train:', train_losses[-1])
                    # restart training op
                    sess.run([self.train_init_op, self.reset_counts])
        print('Molecules observed: ', self.global_steps)
        print('Class proportion: [class] [test] [train]')
        for k,c,ct in zip(self.embedding_dicts['class'].keys(), self.test_counts, self.train_counts):
            print('\t', k, c / sum(self.test_counts), ct / sum(self.train_counts))


    def eval_train(self, feed_dict={}, checkpoint_index=-1):
        predict = []
        out_atoms = []
        out_labels = []
        out_class = []
        out_names = []
        N = 0
        saver = tf.train.Saver()

        with tf.Session() as sess:
            self.load(sess, checkpoint_index)
            if self.using_dataset:
                sess.run(self.test_init_op)
            # do try/except to exhaust test data
            # hope it's not repeated!
            try:
                print('Evaluating test data: ', end='')
                while True:
                    peaks, labels, mask, atoms, class_label, names = sess.run(
                        [self.peaks, self.peak_labels, self.mask, self.atoms, self.class_label, self.names],
                        feed_dict=self._feed_dict(feed_dict, False))
                    for i in range(len(peaks)):
                        for p,l,m,a,n in zip(peaks[i], labels[i], mask[i], atoms[i], names[i]):
                                if m > 0.1:
                                    N += 1
                                    predict.append(p)
                                    out_atoms.append(a)
                                    out_labels.append(l)
                                    out_class.append(class_label[i][0])
                                    out_names.append(n)
                    print('\rEvaluating test data: {}'.format(N), end='')
            except tf.errors.OutOfRangeError:
                print('')
        return predict, out_labels, out_atoms, out_class, out_names

    def eval(self, feed_dict={}):
        saver = tf.train.Saver()
        result = []
        with tf.Session() as sess:
            self.load(sess)
            if self.using_dataset:
                sess.run(self.test_init_op)
            # hope it's not repeated!
            try:
                print('Evaluating test data: ', end='')
                N = 0
                while True:
                    result.append(sess.run(
                        {
                            'peaks': self.peaks,
                            'bonds': self.adjacency,
                            'dist': self.dist_mat,
                            'nlist': self.nlist,
                            'nlist_grad': self.nlist_grad,
                            'mask': self.mask,
                            'names': self.names,
                            'record_index': self.record_index,
                            'atom_embed': self.atom_embed,
                            'bond_embed': self.bond_embed,
                            'atom_embeddings': self.atom_embeddings,
                            'degree_mat': self.degree_mat,
                            'bond_aug': self.bond_aug,
                            'F0': self.feature_mats[0],
                            'F1': self.feature_mats[1],
                            'FL': self.feature_mats[-1]
                        },
                                  feed_dict=self._feed_dict(feed_dict, False)))
                    N += 1
                    print('\rEvaluating test data: {}'.format(N), end='')
            except tf.errors.OutOfRangeError:
                print('')
        return result

    def eval_small(self, feed_dict={}):
        saver = tf.train.Saver()
        result = []
        with tf.Session() as sess:
            self.load(sess)
            if self.using_dataset:
                sess.run(self.test_init_op)
            # hope it's not repeated!
            try:
                print('Evaluating test data: ', end='')
                N = 0
                while True:
                    result.append(sess.run(
                        {
                            'peaks': self.peaks,
                            'class': self.class_label,
                            'mask': self.mask,
                            'peak_labels': self.peak_labels,
                            'names': self.names,
                            'record_index': self.record_index
                        },
                                  feed_dict=self._feed_dict(feed_dict, False)))
                    N += 1
                    print('\rEvaluating test data: {}'.format(N), end='')
            except tf.errors.OutOfRangeError:
                print('')
        return result

    def time_peaks(self, feed_dict={}):
        import timeit
        saver = tf.train.Saver()
        result = []

        with tf.Session() as sess:
            self.load(sess)
            if self.using_dataset:
                sess.run(self.test_init_op)
            # hope it's not repeated!
            try:
                start_time = timeit.default_timer()
                N = 0
                while True:
                    sess.run(
                            self.peaks,
                            feed_dict=self._feed_dict(feed_dict, False))
                    N += self.hypers.BATCH_SIZE
            except tf.errors.OutOfRangeError:
                elapsed = timeit.default_timer() - start_time

        print(f'Elapsed time: {elapsed}. Time per record {elapsed / N}. '
              f'Time per peak {elapsed / N / self.atom_number}')

    def to_networkx(self, number=16, feed_dict = {}):
        import networkx as nx
        if not self.built:
            raise ValueError('Must build first')
        saver = tf.train.Saver()

        peaks, atoms, dist_mat, bonds, features, class_label, mask, labels, record_index = [], [], [], [], [], [], [], [], []

        with tf.Session() as sess:
            self.load(sess)
            while len(peaks) < number:
                if self.using_dataset:
                    sess.run(self.test_init_op)
                    result = sess.run(
                        [
                            self.peaks,
                            self.features,
                            self.dist_mat,
                            self.adjacency,
                            self.feature_mats[-1],
                            self.class_label,
                            self.mask,
                            self.peak_labels,
                            self.record_index
                        ],
                        feed_dict=self._feed_dict(feed_dict, False))
                else:
                    result = sess.run(
                        [
                            self.peaks,
                            self.features,
                            self.dist_mat,
                            self.adjacency,
                            self.feature_mats[-1],
                            self.class_label
                        ],
                        feed_dict=self._feed_dict(feed_dict, False))
                peaks.extend(result[0])
                atoms.extend(result[1])
                dist_mat.extend(result[2])
                bonds.extend(result[3])
                features.extend(result[4])
                class_label.extend(result[5])
                if self.using_dataset:
                    mask.extend(result[6])
                    labels.extend(result[7])
                    record_index.extend(result[7])

        ad = self.embedding_dicts['atom']
        rad = dict(zip(ad.values(), ad.keys()))
        mols = []

        for mi in range(len(peaks)):

            symm_dmat = np.maximum( dist_mat[mi], dist_mat[mi].transpose() )

            name = ''
            for k,v in self.embedding_dicts['class'].items():
                if v == class_label[mi]:
                    name = k
                    break
            g = nx.Graph(class_label=name, record_index=record_index[mi])
            for i,a in enumerate(atoms[mi]):
                if a > 0.1: # check if unit
                    g.add_node(i, name=rad[a], peak=peaks[mi][i],
                               mask = 1 if not self.using_dataset else mask[mi][i],
                               label = None if not self.using_dataset else labels[mi][i],
                               features = ','.join([str(np.round(n, 1)) for n in features[mi][i]]))
            for i in range(len(atoms[mi])):
                for j in range(len(atoms[mi])):
                    if symm_dmat[i, j] > 0.01:
                        g.add_edge(i, j, weight=symm_dmat[i,j], bonded=bonds[mi][i, j] > 0.1)
            mols.append(g)
        return mols

    def build(self, features, adjacency, atom_number):
        raise NotImplementedError()

    def summarize_eval(self, feed_dict={}, test_data=False, plot_dir_name=None, checkpoint_index=-1, classes=True):


        import matplotlib.pyplot as plt
        import os

        predict, labels, atoms, class_label, names = self.eval_train(feed_dict, checkpoint_index)
        predict, labels, atoms, class_label, names = np.array(predict), np.array(labels), np.array(atoms), np.array(class_label), np.array(names)

        # do clipping we do for training
        labels = np.clip(labels, 0, self.hypers.PEAK_CLIP)

        if plot_dir_name is None:
            if not test_data:
                plot_dir = self.model_path + '/plots/'
            else:
                plot_dir = self.model_path + '/plots-test/'
        else:
            plot_dir = os.path.join(self.model_path, plot_dir_name) + '/'
        if checkpoint_index == -1:
            plot_suffix = '.png'
        else:
            plot_suffix = '{:04d}.png'.format(checkpoint_index)
        os.makedirs(plot_dir, exist_ok=True)
        print('Plotting in', plot_dir)
        def plot_fit(fit_labels, fit_predict, fit_class, title):
            print('plotting', title)
            np.savez(plot_dir + title, fit_labels, fit_predict, fit_class)
            rmsd = np.sqrt(np.mean((fit_labels - fit_predict)**2))
            mae = np.mean(np.abs(fit_labels - fit_predict))
            corr = np.corrcoef(fit_labels, fit_predict)[0,1]
            N = len(fit_labels)
            plt.figure(figsize=(5,4))
            if len(np.unique(fit_class) > 5):
                plt.scatter(fit_labels, fit_predict, marker='o', s=6, alpha=0.5, linewidth=0)
            else:
                plt.scatter(fit_labels, fit_predict, marker='o', s=6, alpha=0.5, linewidth=0,
                        c=np.array(fit_class).reshape(-1), cmap=plt.get_cmap('tab20'))
            # take top/bot 1% for upper bound/lower bound (with 20% margin)
            mmin,mmax = np.quantile(fit_labels, q=[0.01, 0.99] )
            mmin,mmax = mmin * 0.8, mmax * 1.2
            plt.plot([0,mmax], [0,mmax], '-', color='gray')
            plt.xlim(mmin, mmax)
            plt.ylim(mmin, mmax)
            plt.xlabel('Measured Shift [ppm]')
            plt.ylabel('Predicted Shift [ppm]')
            #plt.savefig(plot_dir + title + '-nostats' + plot_suffix, dpi=300)
            plt.title(title + ': RMSD = {:.4f}. MAE = {:.4f} R^2 = {:.4f}. N={}'.format(rmsd,mae, corr**2, N), fontdict={'fontsize': 8})
            plt.savefig(plot_dir + title + plot_suffix, dpi=300)
            plt.close()
            return {'corr-coeff': corr, 'R^2': corr**2, 'MAE': mae, 'RMSD': rmsd, 'N': N, 'title': title, 'plot': plot_dir + title + '.png'}

        results = []

        # make overall plot
        results.append(plot_fit(labels, predict, class_label, 'overall'))

        # embedding plots
        for k,v in self.embedding_dicts['atom'].items():
            mask = atoms == v
            if np.sum(mask) == 0:
                continue
            p = predict[mask]
            l = labels[mask]
            c = class_label[mask]
            results.append(plot_fit(l, p, c, f'overall-{k}'))


        # class plots
        if classes:
         os.makedirs(plot_dir + '/class', exist_ok=True)
         for k,v in self.embedding_dicts['class'].items():
             mask = class_label == v
             if np.sum(mask) == 0:
                 continue
             p = predict[mask]
             l = labels[mask]
             c = class_label[mask]
             results.append(plot_fit(l, p, c, 'class/' + k))

         # name plots
         os.makedirs(plot_dir + '/names', exist_ok=True)
         # we want to merge across all residues.
         resnames = {}
         for k,v in self.embedding_dicts['name'].items():
             sp = k.split('-')
             n = sp[0]
             if len(sp) > 1:
                 n = sp[1]
             if n in resnames:
                 resnames[n].append(v)
             else:
                 resnames[n] = [v]
         for k, v in resnames.items():
             mask = names == v[0]
             for vi in v[1:]:
                 mask |= names == vi
             if np.sum(mask) < 10:
                 continue
             p = predict[mask]
             l = labels[mask]
             c = class_label[mask]
             results.append(plot_fit(l, p, c, 'names/' + k))

        plt.figure(figsize=(5,4))
        plt.hist(np.abs(predict - labels), bins=1000)
        plt.savefig(plot_dir + 'peak-error' + plot_suffix, dpi=300)

        # return top 1 error for convienence
        top1 = np.quantile(np.abs(predict - labels), q=[0.99])
        return top1[0], results


    def plot_examples(self, atom_number, cutoff, number, feed_dict={}):
        import networkx as nx
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import pygraphviz_layout

        def fit(g, n):
            if 'mask' not in g.nodes[n]:
                return 0
            if g.nodes[n]['mask'] > 0.1:
                return abs(g.nodes[n]['peak'] - g.nodes[n]['label'])
            return 0
        def color_fit(g, n, cmap, vmin, vmax):
            f = fit(g, n)
            if f == 0:
                return (1,1,1,0)
            else:
                return cmap((f - vmin) / vmax)

        mols = []
        vmax = 0
        candidates = self.to_networkx(number=number * 25)
        for g in candidates:
            for n in g.nodes():
                error = fit(g, n)
                if error > cutoff:
                    if error > vmax:
                        vmax = error
                    mols.append(g)
                    break

        # trim to size
        mols = mols[:number]
        grid = int(np.sqrt(len(mols)))
        fig, axs = plt.subplots(grid, grid, figsize=(32, 32))
        viridis = plt.get_cmap('viridis', 100)
        for i in range(grid):
            for j in range(grid):
                ax = axs[i,j]
                g = mols[i * grid + j]
                r = []
                if atom_number - 1 in g.nodes:
                    g.remove_node(atom_number - 1)
                c = [color_fit(g,n,viridis, 0, vmax) for n in g.nodes]

                # now remove all non single-bond edges
                remove = []
                for e in g.edges(data=True):
                    if not e[2]['bonded']:
                        if 'mask' not in g.nodes[e[1]] or 'mask' not in g.nodes[e[0]]:
                            #TODO how is this happening?
                            remove.append(e)
                        elif g.nodes[e[0]]['mask'] < .1 and g.nodes[e[1]]['mask'] < 0.1:
                            remove.append(e)
                g.remove_edges_from(remove)
                pos = pygraphviz_layout(g, prog='sfdp', args='-Gmaxiter=150 -Gnodesep=100 -Nshape=circle')
                edge_colors = [(0,0,0) if d['bonded'] else (0.8, 0.8, 0.8) for e1,e2,d in g.edges(data=True)]
                #pos = nx.layout.spring_layout(g, iterations=150, k=0.2)
                nx.draw(g,
                pos,
                node_size=60,
                fontsize=5,
                node_color=c,
                vmin=0,
                vmax=vmax,
                labels={n: g.nodes[n]['name'] if'mask' in g.nodes[n] and g.nodes[n]['mask'] < 0.1 else '' for n in g.nodes},
                with_labels=True,
                edge_color=edge_colors,
                ax=ax
                )
                ax.axis('on')
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title(g.graph['class_label'] + ':' + ','.join([str(i) for i in g.graph['record_index']]))
        cm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin = 0, vmax=vmax))
        cb = plt.colorbar(cm, shrink=0.8)
        cb.ax.get_yaxis().labelpad = 15
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_ylabel('Prediction Error', rotation=270, fontsize=16)
        plt.savefig(self.model_path + '/plots/' + 'examples-{:.2f}.jpg'.format(cutoff), dpi=300, transparent=True)
        plt.close()

class StructGCNModel(GCNModel):
    def build(self, features, nlist, atom_number, neighbor_size, predict_atom='H', add_gradients=True):
        self.atom_number = atom_number
        print('Building with ')
        for k,v in self.hypers.__dict__.items():
            print('\t',k, ':', v)
        # no longer print these, too much info
        if False:
         print('\t classes: ', len(self.embedding_dicts['class']))
         print('\t Embeddings: ')
         for k,v in self.embedding_dicts.items():
             print('\t\t ', k)
             for ki, vi in v.items():
                 print('\t\t\t', ki, vi)
        self.features = features
        self.nlist = nlist
        self.training = tf.placeholder(dtype=tf.bool, shape=[])

        if self.using_dataset:
            # filter to train on only a certain atom type
            if predict_atom is not None:
                self.mask = self.raw_mask * tf.cast(tf.math.equal(self.features, self.embedding_dicts['atom'][predict_atom]), tf.float32)
            else:
                self.mask = self.raw_mask
            self.bool_mask = tf.cast(self.mask, tf.bool)

        if self.hypers.EDGE_EMBEDDING_SIZE < 2:
            raise ValueError('Edge embedding must be at least size 2')


        # create learned adjacency matrix
        batch_size = tf.shape(features)[0]
        with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
            self.edge_embeddings = tf.get_variable(
                        name='edge-embedding',
                        initializer=tf.random_uniform(
                            [
                                len(self.embedding_dicts['nlist']),
                                1
                            ],
                            -1, 1))
        with tf.name_scope('edge-preprocess'):
            # pre-process edges according to model type
            # edges are B x N x NN x 3
            # 0 -> distances
            # 1 -> index of neighbor
            # 2 -> type
            flat_edges = tf.reshape(nlist, [batch_size, atom_number * neighbor_size, 3])
            flat_edge_indices = flat_edges[:, :, 1]
            edge_features = tf.nn.embedding_lookup(self.edge_embeddings, tf.cast(flat_edges[:,:,2], tf.int64))

            # remove or select the bond types
            def modify_bond_type(edge_features, flat_edge_indices, bond_type, remove, mod_indices=True):
                mask = tf.cast(tf.math.equal(flat_edges[:,:,2], self.embedding_dicts['nlist'][bond_type]), tf.float32)
                if remove:
                    mask = 1.0 - mask
                # mask is now 1 for stuff I want to keep
                # mask embeded
                # TODO: CHECK THIS this tile dimensions
                # Since I reduced embedding size to 1
                edge_features = tf.tile(tf.reshape(mask, [batch_size, atom_number * neighbor_size, 1]),
                                        [1, 1, 1]) * edge_features
                if mod_indices:
                    # mask indices now
                    # We make those with mask 0 point to special atom at end of molecule
                    flat_edge_indices = flat_edge_indices + (flat_edge_indices - atom_number - 1) * (1.0 - mask)
                return edge_features, flat_edge_indices
            if not self.hypers.EDGE_NONBONDED:
                # Z-DISABLED
                #edge_features, flat_edge_indices = modify_bond_type(edge_features, flat_edge_indices, 'nonbonded', True)
                edge_features, flat_edge_indices = modify_bond_type(edge_features, flat_edge_indices, 'nonbonded', False)
            if not self.hypers.EDGE_LONG_BOND:
                edge_features, flat_edge_indices = modify_bond_type(edge_features, flat_edge_indices,1, False)
            # we must remove all the extra zeros rows
            # so they don't accidentally create interactions with
            # atom 0
            edge_features, flat_edge_indices = modify_bond_type(edge_features, flat_edge_indices, 'none', True, False)

            # convert from nm to angstrom (for plotting/analysis purposes)
            input_distances = flat_edges[:, :, 0] * 10
            tf.summary.histogram('edge-distances', tf.clip_by_value(input_distances, 0.1, 100))
            # distances is now B x (AN * NN)
            if not self.hypers.EDGE_DISTANCE:
                distances = tf.zeros_like(input_distances)
                distances = tf.tile(distances[:,:,tf.newaxis], [1, 1, self.hypers.EDGE_EMBEDDING_SIZE - 1])
            elif self.hypers.EDGE_RBF:
                rbf_expander = RBFExpansion(1.5,8, self.hypers.EDGE_EMBEDDING_SIZE - 1)
                distances = rbf_expander(input_distances)
            else:
                # make them run from 0 (far) to 1 (close)
                distances = tf.clip_by_value(safe_div(1.0, input_distances), 0, 1)
                # tile them to make more
                distances = tf.tile(distances[:,:,tf.newaxis], [1, 1, self.hypers.EDGE_EMBEDDING_SIZE - 1])

            tf.summary.histogram('edge-distances-features', distances[...,0])
            # dimension is batch x (atom_number  * NN) x (edge embed size)
            self.bond_embed = tf.concat([edge_features, distances], axis=2)

            # edge features includes bonded distance and distance. Bonded distance of 0 = intermolecule neighbor
            # total size is edge embedding size
            x = self.bond_embed
            for _ in range(self.hypers.EDGE_FC_LAYERS - 2):
                x = tf.keras.layers.Dense(self.hypers.EDGE_EMBEDDING_SIZE, activation=tf.keras.activations.relu, kernel_regularizer=self.hypers.REGULARIZER)(x)
                if self.hypers.BATCH_NORM:
                    x = tf.keras.layers.BatchNormalization(renorm=True)(x, training=self.training)
            if self.hypers.NON_LINEAR:
                x = tf.keras.layers.Dense(self.hypers.EDGE_EMBEDDING_OUT, activation=tf.keras.activations.tanh, kernel_regularizer=self.hypers.REGULARIZER)(x)
            else:
                x = tf.keras.layers.Dense(self.hypers.EDGE_EMBEDDING_OUT, activation=tf.keras.activations.relu, kernel_regularizer=self.hypers.REGULARIZER)(x)
            self.bond_aug = tf.reshape(x, [batch_size, atom_number, neighbor_size, self.hypers.EDGE_EMBEDDING_OUT])
            if self.using_dataset:
                tf.summary.histogram('bond-aug', tf.boolean_mask(self.bond_aug, self.bool_mask))
            edge_indices = tf.reshape(tf.cast(flat_edge_indices, tf.int32), [batch_size, atom_number, neighbor_size])
            # add batch index
            edge_batch_indices = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, atom_number * neighbor_size]), [batch_size, atom_number, neighbor_size])
            # add embedding index (we get all embeddings)
            # dimension will be B x N x NN x 2 so when we slice
            # we will get B x N x NN x slice
            full_edge_indices = tf.stack([edge_batch_indices, edge_indices], axis=-1)
            # now we have indices and weights for edges


        with tf.name_scope('adj-creation'):
            #this stuff is expensive to do, but this is not part of training or testing
            # we need three indices to slice correctly. So we must add one more to full_edge_indices
            edge_atom_indices = tf.tile(tf.reshape(tf.range(atom_number), [1, -1, 1, 1]), [batch_size, 1, neighbor_size, 1])
            # gather we need -1 x 3, where last shape = [batch_index, a1_index, a2_index]
            adj_edge_indices = tf.concat([full_edge_indices, edge_atom_indices], axis=-1)
            # our index is
            all_adjacency = tf.scatter_nd(tf.reshape(adj_edge_indices, [-1,3]),
                                           tf.reshape(tf.cast(flat_edges[:,:,2], tf.int32), [batch_size * atom_number * neighbor_size]),
                                           [batch_size, atom_number, atom_number])
            # only get things which are single bonded
            asymm_adjacency = tf.cast(tf.math.equal(all_adjacency, self.embedding_dicts['nlist'][1]), tf.int32)
            # make our adjacency matrix symmetric, like one would expect.
            self.adjacency = tf.map_fn(lambda x: tf.math.maximum(x, tf.transpose(x)), asymm_adjacency)
            self.dist_mat = tf.scatter_nd(tf.reshape(adj_edge_indices, [-1,3]),
                                           tf.reshape(flat_edges[:,:,0], [batch_size * atom_number * neighbor_size]),
                                           [batch_size, atom_number, atom_number])

        with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
            self.atom_embeddings = tf.get_variable(
                                name='atom-embedding',
                                initializer=tf.random_uniform(
                                    [
                                        len(self.embedding_dicts['atom']),
                                        self.hypers.ATOM_EMBEDDING_SIZE
                                    ],
                                    -1, 1))
        with tf.name_scope('graph-convs'):
            with tf.name_scope('first-layer'):
                atom_embed = tf.nn.embedding_lookup(self.atom_embeddings, features)
                # make sure atoms that don't exist have an embedding value of 0
                self.atom_embed = atom_embed * tf.cast(features > 0, tf.float32)[:,:,tf.newaxis]
                if self.using_dataset:
                    tf.summary.histogram('atom-embeds', tf.boolean_mask(self.atom_embed, self.bool_mask))
            self.feature_mats = [self.atom_embed]
            self.weights = []

            for l in range(self.hypers.STACKS):
                w = tf.get_variable(
                    'w-{}'.format(l),
                    shape=[self.hypers.ATOM_EMBEDDING_SIZE, self.hypers.ATOM_EMBEDDING_SIZE, self.hypers.EDGE_EMBEDDING_OUT])
                b = tf.get_variable('b-{}'.format(l), shape=[self.hypers.ATOM_EMBEDDING_SIZE])
                tf.summary.histogram(w.name, w)
                # extract feature
                if self.hypers.GCN_DROPOUT:
                    f = tf.keras.layers.Dropout(self.dropout_rate)(self.feature_mats[-1])
                else:
                    f = self.feature_mats[-1]
                sliced_features = tf.gather_nd(f, full_edge_indices)
                # weigh features by edge weights and then go from embedding to embedding
                # create B x N x NN x M via bond_aug and then go from M to M via weights
                prod = tf.einsum('bijn,bijl,lmn->bijm', self.bond_aug, sliced_features, w)
                # now we pool across neighbors
                reduced = tf.reduce_mean(prod, axis=2)
                # activate
                #p1 = tf.print("nlist:", self.nlist, "\n indicies:", full_edge_indices, "\nfeatures:", self.feature_mats[-1], "\nsliced:", sliced_features, summarize=1000)
                #with tf.control_dependencies([p1]):
                if self.hypers.BATCH_NORM:
                    reduced = tf.keras.layers.BatchNormalization(renorm=True)(reduced, training=self.training)
                if self.hypers.GCN_BIAS:
                    out = self.hypers.GCN_ACTIVATION(reduced + b)
                else:
                    out = self.hypers.GCN_ACTIVATION(reduced)
                if self.hypers.RESIDUE:
                    self.feature_mats.append(out + self.feature_mats[-1])
                else:
                    self.feature_mats.append(out)
                self.weights.append(w)
        x = tf.keras.layers.Dropout(self.dropout_rate)(self.feature_mats[-1])
        x0 = x
        for hl in range(self.hypers.FC_LAYERS - 2 if self.hypers.NON_LINEAR else self.hypers.FC_LAYERS - 1):
            x = tf.keras.layers.Dense(self.hypers.ATOM_EMBEDDING_SIZE, activation=None, kernel_regularizer=self.hypers.REGULARIZER)(x)
            if self.hypers.BATCH_NORM:
                x = tf.keras.layers.BatchNormalization(renorm=True)(x, training=self.training)
            if self.hypers.RESIDUE:
                x = x + x0
            x = tf.keras.layers.Dropout(self.dropout_rate, noise_shape=[batch_size, 1, self.hypers.ATOM_EMBEDDING_SIZE])(tf.keras.activations.relu(x))
        # penultimate with non-linearity (?)
        if self.hypers.NON_LINEAR:
            x = tf.keras.layers.Dense(self.hypers.ATOM_EMBEDDING_SIZE // 2, activation=tf.keras.activations.tanh, kernel_regularizer=self.hypers.REGULARIZER)(x)
            if self.hypers.BATCH_NORM:
                x = tf.keras.layers.BatchNormalization(renorm=True)(x, training=self.training)

        # now expand to match range of peaks
        self.peak_std = np.ones(len(self.embedding_dicts['atom']), dtype=np.float32)
        self.peak_avg = np.zeros(len(self.embedding_dicts['atom']), dtype=np.float32)
        for k,v in self.peak_standards.items():
            self.peak_std[k] = v[2]
            self.peak_avg[k] = v[1]

        # output as many feature possibilities as there are
        # then select
        if self.hypers.EMBEDDINGS_OUT:
            dim_size = len(self.embedding_dicts['atom'])
            peak_params = tf.keras.layers.Dense(dim_size, name='feature-dense-out')(x)
            # peak params in B x N x F
            oh = tf.one_hot(self.features, depth=dim_size)
            self.raw_peaks = tf.reduce_sum(peak_params * oh, axis=-1)
            # one hot features in B x N x F
            # peak_std/avg are F
            # the one hot will remove all but one peak, then sum
            self.peaks = tf.reduce_sum(peak_params * oh * self.peak_std + oh * self.peak_avg, axis=-1)
        else:
            self.raw_peaks =  tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(x))
            self.peaks = self.raw_peaks * tf.nn.embedding_lookup(self.peak_std, features) + tf.nn.embedding_lookup(self.peak_avg, features)


        self.built = True

        # to be consistent with other methods, must have something here
        self.degree_mat = x

        tpn = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('TRAINABLE PARAMETERS:', tpn)

            # derivative
            # Will get back a (B x N) of (N x NN)
            # TODO -> you need to decide if you want 
            # to get the ridiculous B x N x N x NN  shape
            # or cut down to non-zero peaks
            # or get a prefactor and reduce to get net
            #dp_dr = tf.map_fn(lambda p: tf.gradients(self.peaks, input_distances)
            #print(x)
            #self.nlist_grad = tf.reshape(x, (batch_size, atom_number, neighbor_size))
        self.nlist_grad = tf.no_op()

        return self.peaks

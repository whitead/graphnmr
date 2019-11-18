from .data import *
import matplotlib.pyplot as plt
import numpy as np

def load_records(filename, batch_size=32):
    data = tf.data.TFRecordDataset(filename, compression_type='GZIP').map(data_parse)
    data = data.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    init_op = iterator.make_initializer(data)
    bond_inputs, atom_inputs, peak_inputs, mask_inputs,name_inputs, class_input = iterator.get_next()
    return init_op, {'features': atom_inputs,
            'nlist': bond_inputs,
            'peaks': peak_inputs,
            'mask': mask_inputs,
            'name': name_inputs,
            'class': class_input}

def peak_summary(data, embeddings, nbins, hist_range, predict_atom='H'):
    mask = data['mask'] * tf.cast(tf.math.equal(data['features'], embeddings['atom'][predict_atom]), tf.float32)
    hist = tf.histogram_fixed_width(data['peaks'] * mask, hist_range, nbins)
    run_ops = []
    # check for nans
    check = tf.check_numerics(data['peaks'], 'peaks invalid')
    run_ops.append(check)
    # throw out zeros
    hist = hist * tf.constant([0] + [1] * (nbins - 1), dtype=tf.int32)
    running_hist = tf.get_variable('peak-hist', initializer=tf.zeros_like(hist), trainable=False)
    run_ops.append(running_hist.assign_add(hist))
    # print out range, suspicious values
    running_min = tf.get_variable('peak-min', initializer=tf.constant(1.))
    running_max = tf.get_variable('peak-max', initializer=tf.constant(1.))
    peaks_min = tf.reduce_min(data['peaks'])
    peaks_max = tf.reduce_max(data['peaks'])
    run_ops.append(running_min.assign(tf.math.minimum(peaks_min, running_min)))
    run_ops.append(running_max.assign(tf.math.maximum(peaks_max, running_max)))
    count = tf.reduce_sum(mask)
    return running_min, running_max, running_hist, count, run_ops

def validate_peaks(filename, embeddings, batch_size=32):
    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)
    nbins = int(1e6)
    hist_range = [0,1e6]
    peaks_min_op, peaks_max_op, histogram_op, count_op, run_ops = peak_summary(data, embeddings, nbins, hist_range)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(init_data_op)
        try:
            i = 0
            while True:
                peaks_min, peaks_max, histogram, count, *_ = sess.run([peaks_min_op, peaks_max_op, histogram_op, count_op] + run_ops)
                i += count
                print('\rValidating Peaks...{} min {} max {}'.format(i, peaks_min, peaks_max), end='')
        except tf.errors.OutOfRangeError:
            print('Dataset complete')
            pass
    print('\nSummary')
    print('N = {}, Min = {}, Max = {}, > {} = {}'.format(i, peaks_min, peaks_max, hist_range[1], histogram[-1]))
    step = nbins / (len(histogram) + 1)
    for i in range(len(histogram)):
        if step * i > 20 and histogram[i] > 0:
            print('Suspicious peaks @ {} (N = {})'.format(step * i, histogram[i]))
    plt.plot(np.arange(0 + step,hist_range[1] - step, step), histogram)
    plt.xlim(0,200)
    plt.savefig('peak-histogram.png', dpi=300)


def validate_embeddings(filename, embeddings, batch_size=32):
    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)
    assert_ops = []
    assert_ops.append(tf.less(tf.reduce_max(data['features']), 
                              tf.constant(max(list(embeddings['atom'].values())), dtype=tf.int64)))
    assert_ops.append(tf.less(tf.reduce_max(tf.cast(data['nlist'][:,:,2], tf.int32)), 
                              tf.constant(max(list(embeddings['nlist'].values())))))
    assert_ops.append(tf.less(tf.reduce_max(data['mask']), 
                              tf.constant(1.)))
    with tf.Session() as sess:
        sess.run(init_data_op)
        try:
            i = 0
            while True:
                sess.run(assert_ops)
                i += 1
                print('\rValidating Embeddings...{}'.format(i), end='')
        except tf.errors.OutOfRangeError:
            pass
    print('\nValid')                

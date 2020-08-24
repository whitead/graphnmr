import tensorflow as tf

def corr_coeff(x, y, w = None):
    if w is None:
        w = tf.ones_like(x)
    m = tf.reduce_sum(w)
    xm = tf.reduce_sum(w * x) / m
    ym = tf.reduce_sum(w * y) / m
    xm2 = tf.reduce_sum(w * x**2) / m
    ym2 = tf.reduce_sum(w * y**2) / m
    cov = tf.reduce_sum( w * (x - xm) * (y - ym) ) / m
    cor = cov / tf.math.sqrt((xm2 - xm**2) * (ym2 - ym**2))
    return cor
    
def corr_loss(labels, predictions, weights, s=0.01):
    '''
    Mostly correlation, with small squared diff
    '''
    x = predictions
    y = labels
    l2 = tf.reduce_sum( weights * ( y - x)**2 ) / tf.reduce_sum(weights)
    return s * l2 + (1 - corr_coeff(x, y, weights))
    

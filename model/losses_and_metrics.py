import os, sys
sys.path.append(os.getcwd())

from assets.utils import K, tf


def euclideanDistance(v):
    x, y = v
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def cosine_distance(v):
    x, y = v
    x = tf.nn.l2_normalize(x, axis=1)
    y = tf.nn.l2_normalize(y, axis=1)
    return tf.constant([1.],shape=(1,1)) -K.sum(x * y, axis=1, keepdims=True)

def dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_original, y_pred):
    sqaure_pred = K.square(y_pred)
    margin = 1
    margin_square = K.square(K.maximum(0, margin - y_pred))
    return K.mean(y_original * sqaure_pred + (1 - y_original) * margin_square)

def accuracy(y_original, y_pred):
    return K.mean(K.equal(y_original, K.cast(y_pred < 0.5, y_original.dtype)))
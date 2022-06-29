"""
Loss and metrics for the SimpleNN model
"""

import tensorflow as tf
import keras.backend as K


def euclidean_distance(_v):
    """
    Computes the euclidean distance between two vectors
    """
    _x, _y = _v
    sum_square = K.sum(K.square(_x - _y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def cosine_distance(_v):
    """
    Computes the cosine distance between two vectors
    """
    _x, _y = _v
    _x = tf.nn.l2_normalize(_x, axis=1)
    _y = tf.nn.l2_normalize(_y, axis=1)
    return tf.constant([1.],shape=(1,1)) -K.sum(_x * _y, axis=1, keepdims=True)

def dist_output_shape(shapes):
    """
    Computes the output shape of the distance layer
    """
    shape1, _ = shapes
    return (shape1[0], 1)

def contrastive_loss(y_original, y_pred):
    """
    Computes the contrastive loss
    """
    sqaure_pred = K.square(y_pred)
    margin = 1
    margin_square = K.square(K.maximum(0, margin - y_pred))
    return K.mean(y_original * sqaure_pred + (1 - y_original) * margin_square)

def accuracy(y_original, y_pred):
    """
    Computes the accuracy
    """
    return K.mean(K.equal(y_original, K.cast(y_pred < 0.5, y_original.dtype)))

# EOL

"""
Helper function to create a model with a single layer.
"""
import random
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def eucledian_distance(_x, _y):
    """
    Calculate the eucledian distance between two vectors
    """
    squared_sum =  K.sqrt(K.sum(K.square(_x - _y)), axis=1, keep_dims=True)
    distance = K.sqrt(K.maximum(squared_sum, K.epsilon()))
    return distance

def eucledian_output_shape(_x, _y):
    """
    Calculate the eucledian distance between two vectors
    """
    assert _x.shape[0] == _y.shape[0], "embedding sizes must be equal"
    return (_x.shape[0], 1)

def accuracy(y_original, y_pred, threshold=0.5):
    """
    Calculate the accuracy of a model
    """
    return K.mean(K.equal(y_original, K.cast(y_pred < threshold, y_original.dtype)))


# pairs

class Pair():
    """
    def make_pairs(self, num_classes):
    """
    def __init__(self,data):
        self._x, self._y = data

    def get_pairs(self):
        """
        Get pairs of images
        """
        _x, _y = self._x, self._y
        pairs, labels = self.make_pairs(len(np.unique(_y)))
        return ImageDataGenerator.flow(pairs, labels)

    def make_pairs(self, num_classes):
        """
        Make pairs of images
        """
        _x, _y = self._x, self._y
        digit_indices = [np.where(_y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        # Iterating and creating positive pairs and negative pairs
        for idx1,_ in enumerate(_x):
            x_1 = _x[idx1]
            label1 = _y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x_2 = _x[idx2]

            labels += list([1])
            pairs += [[x_1, x_2]]

            label2 = random.randint(0, num_classes-1)
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx2 = random.choice(digit_indices[label2])
            x_2 = _x[idx2]

            # Generate negative pair labels for the same
            labels += list([0])
            pairs += [[x_1, x_2]]

        return np.array(pairs), np.array(labels)


# EOL

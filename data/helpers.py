"""
Helper Functions
"""
import random
import numpy as np
import tensorflow as tf

# pairs
class Pair():
    """
    def make_pairs(self, num_classes):
    """
    def __init__(self, data):
        _x, _y = data
        self._x, self._y = np.array(_x), np.array(_y)

    def decode_img(self, img):
        """
        Decode the image from the path.
        """
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def get_pairs(self):
        """
        Get the pairs of images and labels.
        """
        _, _y = self._x, self._y
        pairs, labels = self.make_pairs(len(np.unique(_y)))
        element_1 = tf.data.Dataset.from_tensor_slices(pairs[:, 0])
        element_2 = tf.data.Dataset.from_tensor_slices(pairs[:, 1])
        labels = tf.data.Dataset.from_tensor_slices(labels)
        return (element_1, element_2, labels)

    def make_pairs(self, num_classes):
        """
        Make pairs of images and labels.
        """
        _x, _y = self._x, self._y
        digit_indices = [np.where(_y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        # Positive and Negative pairs
        for idx_1,_ in enumerate(_x):
            x_1 = _x[idx_1]
            label1 = _y[idx_1]
            idx_2 = random.choice(digit_indices[label1])
            x_2 = _x[idx_2]

            labels += list([1])
            pairs += [[x_1, x_2]]

            label2 = random.randint(0, num_classes-1)
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx_2 = random.choice(digit_indices[label2])
            x_2 = _x[idx_2]

            labels += list([0])
            pairs += [[x_1, x_2]]

        return np.array(pairs), np.array(labels)

# EOF

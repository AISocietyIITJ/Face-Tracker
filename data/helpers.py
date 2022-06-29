"""
Some Helper Functions
"""
import random
import numpy as np
import tensorflow as tf


# pairs
class Pair(object):
    """
    def makePairs(self, num_classes):
    """
    def __init__(self,data):
        _x, _y = data
        self._x, self._y = np.array(_x), np.array(_y)

    def decode_img(self, img):
        """
        Decode the image into a numpy array
        """
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def get_pairs(self):
        """
        Get the pairs
        """
        _x, _y = self._x, self._y
        print(_x,_y)
        pairs, labels = self.make_pairs(len(np.unique(_y)))
        element_1 = tf.data.Dataset.from_tensor_slices(pairs[:, 0])
        element_2 = tf.data.Dataset.from_tensor_slices(pairs[:, 1])
        labels = tf.data.Dataset.from_tensor_slices(labels)
        return (element_1, element_2, labels)

    def make_pairs(self, num_classes):
        """
        Make the pairs
        """
        _x, _y = self._x, self._y
        digit_indices = [np.where(_y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        for idx1, _ in enumerate(_x):
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

            labels += list([0])
            pairs += [[x_1, x_2]]

        return np.array(pairs), np.array(labels)

class Augment():
    """
    def augment_pairs(self, element_set_1, element_set_2, pair_labels, augment_config):
    """
    def rotate_img(self, img):
        """
        Rotate the image by a random angle between -10 and 10 degrees.
        """
        img = tf.keras.layers.RandomRotation(0.2)(img)
        return img

    def zoom_img(self, img):
        """
        Zoom the image by a random factor between 0.8 and 1.2.
        """
        img = tf.keras.layers.RandomZoom(0.5)(img)
        return img

    def shift_img(self, img):
        """
        Shift the image by a random factor between -0.1 and 0.1.
        """
        img = tf.keras.layers.RandomShift(0.5)(img)
        return img

    def flip_img(self, img):
        """
        Flip the image horizontally.
        """
        img = tf.keras.layers.RandomFlip()(img)
        return img

    def shear_img(self, img):
        """
        Shear the image by a random factor between -0.1 and 0.1.
        """
        img = tf.keras.preprocessing.image.random_shear(img, 0.2)
        return img


def augment_pairs(image_data_1,image_data_2,labels,augmentation_config):
    """
    Augment the pairs
    """

    augmented_image_data_1 = image_data_1
    augmented_image_data_2 = image_data_2
    augmented_labels = labels

    augment = Augment()
    if "rotation_range" in augmentation_config:
        rotated_1 = image_data_1.map(augment.rotate_img)
        rotated_2 = image_data_2.map(augment.rotate_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(rotated_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(rotated_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "width_shift_range" in augmentation_config:
        w_shifted_1 = image_data_1.map(augment.shift_img)
        w_shifted_2 = image_data_2.map(augment.shift_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(w_shifted_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(w_shifted_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "height_shift_range" in augmentation_config:
        h_shifted_1 = image_data_1.map(augment.shift_img)
        h_shifted_2 = image_data_2.map(augment.shift_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(h_shifted_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(h_shifted_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "zoom_range" in augmentation_config:
        zoomed_1 = image_data_1.map(augment.zoom_img)
        zoomed_2 = image_data_2.map(augment.zoom_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(zoomed_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(zoomed_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "flip_horizontal" in augmentation_config:
        flipped_1 = image_data_1.map(augment.flip_img)
        flipped_2 = image_data_2.map(augment.flip_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(flipped_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(flipped_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "shear_range" in augmentation_config:
        sheared_1 = image_data_1.map(Augment.shear_img)
        sheared_2 = image_data_2.map(Augment.shear_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(sheared_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(sheared_2)
        augmented_labels = augmented_labels.concatenate(labels)

    return (augmented_image_data_1,augmented_image_data_2,augmented_labels)


# EOL

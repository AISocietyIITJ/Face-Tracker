"""
Generating the Dataset (Memory Effecient)
"""
import os
import sys
import json
import tensorflow as tf
import tensorflow.kerras.preprocessing.image as ImageDataGenerator
sys.path.append("../")
from data.helpers import Pair, augment_pairs

with open("encodings.json", encoding="utf-8") as load_value:
    config = json.load(load_value)

def generate_data_for_classifier(data_dir, image_dir, is_augmented=False):
    """
    Generate data for classifier
    """
    images = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=config{"shear_range"},
        zoom_range=config{"zoom_range"},
        horizontal_flip=config{"horizontal_flip"},
        rotation_range=config{"rotation_range"},
    )

    data = images.flow_from_directory(f"{data_dir}/{image_dir}")

    return data


def generate_data_for_siamese(data_dir, image_dir, is_augmented=False):
    """
    Generate data for siamese network
    """

    # data_dir = 'data'
    # image_dir = f'{data_dir}\\gestures'

    images = []
    labels = []

    # Map paths to images

    def decode_img(img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        # img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    for folder in os.listdir(image_dir):
        for image in os.listdir(f'{image_dir}/{folder}'):
            images.append(f'{image_dir}/{folder}/{image}')
            labels.append(int(folder))

    pair_generator = Pair((images, labels))
    element_set_1, element_set_2, pair_labels =  pair_generator.get_pairs()

    # Evaluate the dataset

    element_set_1 = element_set_1.map(decode_img)
    element_set_2 = element_set_2.map(decode_img)
    pair_labels = pair_labels.map(lambda x: tf.cast(x, tf.int64))

    # Augment the dataset

    if is_augmented:
        with open(
            data_dir + '/' + "augmentations.json", encoding="utf-8") as augmentation:
            augment_config = json.load(augmentation)

        element_set_1, element_set_2, pair_labels = augment_pairs(
            element_set_1,
            element_set_2,
            pair_labels,
            augment_config
            )

    data =  tf.data.Dataset.zip((element_set_1, element_set_2), pair_labels)
    data = data.shuffle(buffer_size=len(images))
    data = data.batch(128)

    return data


# EOL

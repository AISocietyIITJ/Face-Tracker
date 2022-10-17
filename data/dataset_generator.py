"""
Dataset generator for the siamese network.
"""
import os
import json
import sys
import random
import tensorflow as tf
from PIL import Image
sys.path.append('../')
from data.helpers import Pair

with open('model/model_specifications.json', encoding='utf-8') as load_data:
    config = json.load(load_data)

with open('data/encodings.json', encoding='utf-8') as load_value:
    img_enc = json.load(load_value)

def generate_data_for_classifier(image_dir, is_augmented=False, batch_size=128):
    """
    Generate data for the classifier.
    """

    images = []
    labels = []

    # Map paths to images
    for foldername in os.listdir(image_dir + '/'):
        for filename in os.listdir(image_dir + '/' + foldername + '/'):
            if filename.endswith('.jpg'):
                try:
                    img = Image.open(image_dir + '/' + foldername + '/' + filename)
                    img.verify()

                    # image path
                    images.append(image_dir + '/' + foldername + '/' + filename)

                    # corresponding label to the image path indicating the folder name
                    labels.append(foldername)
                    
                except (IOError, SyntaxError) as e:
                    pass

    def decode_img(img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (config["HEIGHT"], config["WIDTH"]))
        return img

    collection = list(zip(images, labels))
    random.shuffle(collection)
    images, labels = zip(*collection)
    images = list(images)
    labels = list(labels)

    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(decode_img)
    labels = tf.one_hot(labels, config["num_classes"])
    labels = tf.data.Dataset.from_tensor_slices(labels)
    labels = labels.map(lambda x: tf.cast(x, tf.int64))
    if is_augmented:
        pass

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.batch(batch_size)

    return dataset


def generate_data_for_siamese(image_dir, is_augmented=False, batch_size=128):
    """
    Generate data for the siamese network.
    """
    images = []
    labels = []

    # Map paths to images
    for foldername in os.listdir(image_dir + '/'):
        for filename in os.listdir(image_dir + '/' + foldername + '/'):
            if filename.endswith('.jpg'):
                try:
                    img = Image.open(image_dir + '/' + foldername + '/' + filename)
                    img.verify()

                    # image path
                    images.append(image_dir + '/' + foldername + '/' + filename)

                    # corresponding label to the image path indicating the folder name
                    labels.append(foldername)
                    
                except (IOError, SyntaxError) as e:
                    pass

    def decode_img(img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (config["HEIGHT"], config["WIDTH"]))
        return img

    pair_generator = Pair((images, labels))
    element_set_1, element_set_2, pair_labels =  pair_generator.get_pairs()

    element_set_1 = element_set_1.map(decode_img)
    element_set_2 = element_set_2.map(decode_img)
    data = tf.data.Dataset.zip(((element_set_1, element_set_2), pair_labels))
    data = data.shuffle(3).batch(batch_size)

    return data

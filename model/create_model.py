"""
Dataset creation tools
"""

import sys
import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend as K
import tennsorflow.keras.applications as applications

sys.path.append("../")
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, Input, Model
from model.losses_and_metrics import euclidean_distance, dist_output_shape, cosine_distance, contrastive_loss

def get_base(base_architecture, config, Input, isTrainable=True):
    """
    Get the base architecture
    """

    if base_architecture == 'VGG16':
        base_model = applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(config["Height"], config["WIDTH"], 3),
        )

    elif base_architecture == 'VGG19':
        base_model = applications.vgg19.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(config["Height"],
            config["WIDTH"], 3),
            )

    elif base_architecture == 'ResNet50':
        base_model = applications.resnet50.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(config["HEIGHT"], config["WIDTH"], 3),
            )

    elif base_architecture == 'InceptionV3':
        base_model = applications.inception_resnet_v2.InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(config["HEIGHT"], config["WIDTH"], 3),
            )

    elif base_architecture == 'efficientnetb0':
        base_model = applications.efficientnet.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(config["HEIGHT"], config["WIDTH"], 3),
            )

    for layer in base_model.layers:
        layer.trainable = isTrainable

    out = base_model(Input)
    return out


def get_siamese_loss(loss_type):
    """
    Get the loss function for the siamese network
    """

    if loss_type == 'contrastive':
        loss = contrastive_loss

    elif loss_type == 'triplet_batch_hard':
        loss = tfa.losses.TripletHardLoss()

    elif loss_type == 'batch_semi_hard':
        loss = tfa.losses.TripletSemiHardLoss()

    return loss


def get_classifier_loss(loss_type):
    """
    Get the loss function for the classifier network
    """

    if loss_type == 'categorical_crossentropy':
        loss = tf.keras.losses.CategoricalCrossentropy()

    elif loss_type == 'sparse_categorical_crossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    elif loss_type == 'binary_crossentropy':
        loss = tf.keras.losses.BinaryCrossentropy()

    elif loss_type == 'categorical_hinge':
        loss = tf.keras.losses.CategoricalHinge()

    elif loss_type == 'hinge':
        loss = tf.keras.losses.Hinge()

    elif loss_type == 'mean_absolute_percentage_error':
        loss = tf.keras.losses.MeanAbsolutePercentageError()

    elif loss_type == 'mean_squared_logarithmic_error':
        loss = tf.keras.losses.MeanSquaredLogarithmicError()

    return loss


def get_model(config):
    """
    Get the siamese network
    """
    base_architecture = config["base_architecture"]
    model_type = config["model_type"]

    INPUT = Input(shape=(config["HEIGHT"], config["WIDTH"], 3))
    base = get_base(base_architecture, config, INPUT)

    if model_type == 'siamese':
        layer = Flatten()(base)
        layer = Dense(1024, activation='relu')(layer)
        layer = Dropout(0.2)(layer)
        # layer = Dense(512, activation='relu')(layer)
        # layer = Dropout(0.2)(layer)
        layer = Dense(config["embedding_size"])(layer)
        out = Lambda(lambda  x: K.l2_normalize(x,axis=1))(layer)
        embedding = Model(INPUT, out)

        Input_1 = Input(shape=(config["HEIGHT"], config["WIDTH"], 3))
        Input_2 = Input(shape=(config["HEIGHT"], config["WIDTH"], 3))

        embedding_1 = embedding(Input_1)
        embedding_2 = embedding(Input_2)

        distance = Lambda(
            euclidean_distance,
            output_shape=dist_output_shape)([embedding_1, embedding_2])

        model = Model([Input_1, Input_2], distance)

        loss = get_siamese_loss(config["siamese_loss"])


    model.compile(loss=loss, optimizer=config["optimizer"], metrics=['accuracy'])

    return model


# EOL

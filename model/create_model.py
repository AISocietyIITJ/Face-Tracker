"""
Creating base model
"""
import sys
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Dropout
from tensorflow.keras.models import Model, Sequential
sys.path.append("../")
from model.helpers import eucledian_distance, eucledian_output_shape



def get_siamese_loss(loss_type):
    """
    Loss function for siamese network
    """
    if loss_type == 'contrastive_loss':
        loss = tfa.losses.ContrastiveLoss(margin=1)
    elif loss_type == 'triplet_hard_loss':
        loss = tfa.losses.TripletHardLoss(margin=1)
    elif loss_type == 'triplet_semihard_loss':
        loss = tfa.losses.TripletSemiHardLoss(margin=1)
    return loss

def get_classifier_loss(loss_type):
    """
    Loss function for classifier network
    """
    if loss_type == 'categorical_crossentropy':
        loss = 'categorical_crossentropy'
    elif loss_type == 'mse':
        loss = 'mse'
    return loss


def get_model(model_type, num_classes, config):
    """
    This function is used to create the model.
    """
    input_tensor = Input(shape=(config["HEIGHT"], config["WIDTH"], 3))

    base = Dense(64, input_shape = (63,), activation='relu')(input_tensor)
    base = Dense(64, activation='relu')(base)
    base = Dense(32, activation='relu')(base)
    base = Dense(16, activation='relu')(base)


    # Defining the model tail

    if model_type == 'siamese':
        layer = Dense(16, activation='relu')(base)
        layer = Dropout(0.2)(layer)
        layer = Dense(8)(layer)
        out = Lambda(lambda  x: K.l2_normalize(x,axis=1))(layer)
        embedding = Model(input_tensor, out)

        input_1 = Input(shape=(63,))
        input_2 = Input(shape=(63,))

        embedding_1 = embedding(input_1)
        embedding_2 = embedding(input_2)

        distance = Lambda(
            eucledian_distance,
            output_shape=eucledian_output_shape)(embedding_1, embedding_2)

        model = Model([input_1, input_2], distance)

        loss = get_siamese_loss(config["loss"])

    elif model_type == 'classifier':
        layer = Dense(8, activation='relu')(base)
        layer = Dense(num_classes, activation='softmax')(layer)
        loss = config["loss"]
        model = Model(input_tensor, layer)

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model

# EOL

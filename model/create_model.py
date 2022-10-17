"""
model functions
"""
import sys
import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend as K
import tensorflow.keras.applications as applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, Multiply, AveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, UpSampling2D
sys.path.append('../')
from model.losses_and_metrics import euclidean_distance, cosine_distance
from model.losses_and_metrics import dist_output_shape, contrastive_loss


def get_base(base_architecture, config, input_tensor, is_trainable=True):

    """
    Returns the base model for the architecture.
    """

    if base_architecture == 'vgg16':
        base_model = applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(config["Height"], config["WIDTH"], 3),
            )

    elif base_architecture == 'vgg19':
        base_model = applications.vgg19.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(config["Height"],
            config["WIDTH"], 3),
            )

    elif base_architecture == 'resnet50':
        base_model = applications.resnet50.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(config["HEIGHT"], config["WIDTH"], 3),
            )

    elif base_architecture == 'inceptionv3':
        base_model = applications.inception_resnet_v2.InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(config["HEIGHT"], config["WIDTH"], 3),
            )

    elif base_architecture == 'efficientnetb5':
        base_model = applications.efficientnet.EfficientNetB5(
            weights='imagenet',
            include_top=False,
            input_shape=(config["HEIGHT"], config["WIDTH"], 3),
            )

    input_base = base_model.input
    output_base = base_model.output
    base = Model(inputs=input_base, outputs=output_base)
    for idx,layer in enumerate(base.layers[::-1]):
        if idx < 16:
            layer.trainable = is_trainable
        else:
            layer.trainable = False
    out = base(input_tensor)
    return out

def get_siamese_loss(loss_type):
    """
    Returns the loss function for the siamese network.
    """
    loss_type = loss_type.lower()
    if loss_type == 'contrastive':
        loss = contrastive_loss

    elif loss_type == 'triplet_batch_hard':
        loss = tfa.losses.TripletHardLoss()

    elif loss_type == 'batch_semi_hard':
        loss = tfa.losses.TripletSemiHardLoss()

    return loss


def get_classifier_loss(loss_type):
    """
    Returns the loss function for the classifier network.
    """
    loss_type = loss_type.lower()
    if loss_type == 'categorical_crossentropy':
        loss = tf.keras.losses.CategoricalCrossentropy()

    elif loss_type == 'sparse_categorical_crossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    elif loss_type == 'binary_crossentropy':
        loss = tf.keras.losses.BinaryCrossentropy()

    elif loss_type == 'categorical_hinge':
        loss = tf.keras.losses.CategoricalHinge()

    elif loss_type == 'weighted_binary_crossentropy':
        def create_weighted_binary_crossentropy(zero_weight, one_weight):
            def weighted_binary_crossentropy(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                b_ce = K.binary_crossentropy(y_true, y_pred)
                weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
                weighted_b_ce = weight_vector * b_ce
                return K.mean(weighted_b_ce)

            return weighted_binary_crossentropy

        loss = create_weighted_binary_crossentropy(zero_weight=1, one_weight=15)
        return loss


def get_model(config):
    """
    Returns the model for the architecture.
    """

    base_architecture = config["base_architecture"]
    input_tensor = Input(shape=(config["HEIGHT"], config["WIDTH"], 3))
    base = get_base(base_architecture, config, input_tensor)
    num_classes = config["num_classes"]

    # Building Classifier Model with Residual Attention Network Architecture
    if config["model_type"].lower() == 'ran_classifier':
        model = ResidualAttentionNetwork(
        base=base,
        input_shape=(config["HEIGHT"], config["WIDTH"], 3),
        n_classes=num_classes,
        activation='softmax'
        ).build_model()

        loss = get_classifier_loss(config["loss"])

    # Building Similarity Network Model
    elif config["model_type"].lower() == 'siamese':
        layer = Flatten()(base)
        layer = Dense(config["embedding_size"]*2, activation='relu')(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(config["embedding_size"],activation = 'softmax')(layer)
        layer = Lambda(lambda  x: K.l2_normalize(x,axis=1))(layer)

        embedding = Model(input_tensor, layer)

        input_1 = Input(shape=(config["HEIGHT"], config["WIDTH"], 3))
        input_2 = Input(shape=(config["HEIGHT"], config["WIDTH"], 3))

        embedding_1 = embedding(input_1)
        embedding_2 = embedding(input_2)

        distance = Lambda(
            euclidean_distance,
            output_shape=dist_output_shape)([embedding_1, embedding_2])
        model = Model([input_1, input_2], distance)

        loss = get_siamese_loss(config["loss"])

    # Buliding Classifier
    elif config["model_type"].lower() == 'classifier':
        layer = Flatten()(base)
        layer = Dense(256, activation='relu')(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(128, activation='relu')(layer)
        out = Dense(num_classes, activation='softmax')(layer)

        loss = get_classifier_loss(config["loss"])

        model = Model(input_tensor, out)

    model.compile(loss=loss, optimizer=config["optimizer"], metrics=['accuracy'])
    return model

# Code for the Residual Attention Network Architecture
class ResidualAttentionNetwork():
    """
    Residual attention classifier networks
    """
    def __init__(self, base, input_shape, n_classes, activation, _p=1, _t=2, _r=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = activation
        self._p = _p
        self._t = _t
        self._r = _r
        self.base = base

    def build_model(self):
        """
        Builds the model
        """
        # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)

        # Initial Layers before Attention Module

        preprocessed_input = tf.keras.applications.efficientnet.preprocess_input(input_data)

        # Initialize base model for generating a feature map
        model = self.base

        for idx, layer in enumerate(model.layers[::-1]):
            if idx < 50:
                layer.trainable = True
            else:
                layer.trainable = False
        _layer = model(preprocessed_input)

        # Residual Unit then Attention Module #1
        res_unit_1 = self.residual_unit(_layer, filters=[32, 64, 128])
        att_mod_1 = self.attention_module(res_unit_1, filters=[32, 64, 128])

        # Residual Unit then Attention Module #2
        res_unit_2 = self.residual_unit(att_mod_1, filters=[32, 64, 128])
        att_mod_2 = self.attention_module(res_unit_2, filters=[32, 64, 128])

        # Residual Unit then Attention Module #3
        res_unit_3 = self.residual_unit(att_mod_2, filters=[32, 64, 128])
        att_mod_3 = self.attention_module(res_unit_3, filters=[32, 64, 128])

        # Ending it all
        res_unit_end_1 = self.residual_unit(att_mod_3, filters=[32, 32, 64])
        res_unit_end_2 = self.residual_unit(res_unit_end_1, filters=[32, 32, 64])
        res_unit_end_3 = self.residual_unit(res_unit_end_2, filters=[32, 32, 64])
        res_unit_end_4 = self.residual_unit(res_unit_end_3, filters=[32, 32, 64])

        # Avg Pooling
        avg_pool_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(res_unit_end_4)

        # Flatten the data
        flatten_op = Flatten()(avg_pool_layer)

        # FC Layers for prediction
        fully_connected_layer_1 = Dense(256, activation='relu')(flatten_op)
        dropout_layer_1 = Dropout(0.5)(fully_connected_layer_1)
        fully_connected_layer_2 = Dense(256, activation='relu')(dropout_layer_1)
        dropout_layer_2 = Dropout(0.5)(fully_connected_layer_2)
        fully_connected_layer_3 = Dense(256, activation='relu')(dropout_layer_2)
        dropout_layer_3 = Dropout(0.5)(fully_connected_layer_3)
        fully_connected_layer_last = Dense(
            self.n_classes, activation=self.activation)(dropout_layer_3)

        # Fully constructed model
        model = Model(inputs=input_data, outputs=fully_connected_layer_last)

        return model

    # Pre-Activation Identity ResUnit Bottleneck Architecture
    def residual_unit(self, residual_input_data, filters):
        """
        Residual Unit Bottleneck Architecture
        """

        # Hold input_x here for later processing
        identity_x = residual_input_data

        # Layer 1
        batch_norm_op_1 = BatchNormalization()(residual_input_data)
        activation_op_1 = Activation('relu')(batch_norm_op_1)
        conv_op_1 = Conv2D(filters=filters[0],
                        kernel_size=(1,1),
                        strides=(1,1),
                        padding='same')(activation_op_1)

        # Layer 2
        batch_norm_op_2 = BatchNormalization()(conv_op_1)
        activation_op_2 = Activation('relu')(batch_norm_op_2)
        conv_op_2 = Conv2D(filters=filters[1],
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding='same')(activation_op_2)

        # Layer 3
        batch_norm_op_3 = BatchNormalization()(conv_op_2)
        activation_op_3 = Activation('relu')(batch_norm_op_3)
        conv_op_3 = Conv2D(filters=filters[2],
                        kernel_size=(1,1),
                        strides=(1,1),
                        padding='same')(activation_op_3)

        # Element-wise Addition
        if identity_x.shape[-1] != conv_op_3.shape[-1]:
            filter_n = conv_op_3.shape[-1]

            identity_x = Conv2D(filters=filter_n,
                            kernel_size=(1,1),
                            strides=(1,1),
                            padding='same')(identity_x)

        output = Add()([identity_x, conv_op_3])

        return output

    def attention_module(self, attention_input_data, filters):
        """
        Attention Module
        """
        # Send input_x through #_p residual_units
        p_res_unit_op_1 = attention_input_data
        for _ in range(self._p):
            p_res_unit_op_1 = self.residual_unit(p_res_unit_op_1, filters=filters)

        # Perform Trunk Branch Operation
        trunk_branch_op = self.trunk_branch(
            trunk_input_data=p_res_unit_op_1, filters=filters)

        # Perform Mask Branch Operation
        mask_branch_op = self.mask_branch(
            mask_input_data=p_res_unit_op_1, filters=filters)

        # Perform Attention Residual Learning: Combine Trunk and Mask branch results
        ar_learning_op = self.attention_residual_learning(
            mask_input=mask_branch_op, trunk_input=trunk_branch_op)

        # Send branch results through #_p residual_units
        p_res_unit_op_2 = ar_learning_op
        for _ in range(self._p):
            p_res_unit_op_2 = self.residual_unit(p_res_unit_op_2, filters=filters)

        return p_res_unit_op_2

    def trunk_branch(self, trunk_input_data, filters):
        """
        Trunk Branch
        """
        # sequence of residual units, default=2
        t_res_unit_op = trunk_input_data
        for _ in range(self._t):
            t_res_unit_op = self.residual_unit(t_res_unit_op, filters=filters)

        return t_res_unit_op

    def mask_branch(self, mask_input_data, filters, _m=3):
        """
        Mask Branch
        """

        downsampling = MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same')(mask_input_data)

        for _ in range(_m):

            for _ in range(self._r):
                downsampling = self.residual_unit(
                    residual_input_data=downsampling, filters=filters)

            downsampling = MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='same')(downsampling)

        # Middle Residuals - Perform 2*_r residual units steps before upsampling
        middleware = downsampling
        for _ in range(2 * self._r):
            middleware = self.residual_unit(residual_input_data=middleware, filters=filters)

        # Upsampling Step Initialization - Top
        upsampling = UpSampling2D(size=(2, 2))(middleware)

        for _ in range(_m):
            # Perform residual units ops _r times between adjacent pooling layers
            for _ in range(self._r):
                upsampling = self.residual_unit(
                    residual_input_data=upsampling, filters=filters)

            # Last interpolation step - Bottom
            upsampling = UpSampling2D(size=(2, 2))(upsampling)

        conv_filter = upsampling.shape[-1]
        conv1 = Conv2D(filters=conv_filter,
                        kernel_size=(2,2),
                        strides=(1,1),
                        padding='same')(upsampling)


        conv2 = Conv2D(filters=conv_filter,
                        kernel_size=(2,2),
                        strides=(1,1),
                        padding='valid')(conv1)

        sigmoid = Activation('sigmoid')(conv2)

        return sigmoid

    def attention_residual_learning(self, mask_input, trunk_input):
        """
        Attention Residual Learning
        """

        m_x = Lambda(lambda x: 1 + x)(mask_input) # 1 + mask
        return Multiply()([m_x, trunk_input]) # M(x) * T(x)

# EOF

"""
Main training function
"""
from datetime import datetime
import json
import tensorflow as tf
from model.create_model import get_model
from data.dataset_generator import generate_data_for_classifier, generate_data_for_siamese
from tensorflow.keras.callbacks import CSVLogger

with open('configurations.json', encoding='utf-8') as load_value:
    configurations = json.load(load_value)

with open(
    configurations["MODEL_DIR"] + "/" + "model_specifications.json", encoding='utf-8') as specs:
    model_config = json.load(specs)

def main(_args):
    """
    Main function for training the model.
    """

    # Loading the Data
    if model_config["model_type"].lower() == "siamese":
        data = generate_data_for_siamese(
            configurations["IMAGE_DIR"],
            is_augmented=False,
            batch_size=configurations["BATCH_SIZE"]
        )

    elif model_config["model_type"].lower() == "classifier":
        data = generate_data_for_classifier(
            configurations["IMAGE_DIR"],
            is_augmented=False,
            batch_size=configurations["BATCH_SIZE"]
        )

    elif model_config["model_type"].lower() == "ran_classifier":
        data = generate_data_for_classifier(
            configurations["IMAGE_DIR"],
            is_augmented=False,
            batch_size=configurations["BATCH_SIZE"]
        )

    elif model_config["model_type"].lower() == "save":
        data = generate_data_for_classifier(
            configurations["IMAGE_DIR"],
            is_augmented=False,
            batch_size=configurations["BATCH_SIZE"],
        )
        return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=configurations["GPU_MEMORY_LIMIT"])]
                        ) # Allocating 37 GB of memory out of 40GB available

                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as _e:
                # Virtual devices must be set before GPUs have been initialized
                print(_e)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        # Loading the model
        model = get_model(model_config)

        # saving the history of the model
        csv_log = CSVLogger(
            "../history/"+
            str(model_config["model_type"])+
            str(model_config["base_architecture"])+
            str(model_config["__loss_options__"]["siamese_loss"])+
            str(model_config["optimizer"])+
            "-training.log",
            separator=",",
            append=True
            )

        # saving model checkpoints
        weights_dir = "../weights"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            weights_dir +
            "/"+
            str(model_config["model_type"])+"-{epoch}-"+str(model_config["base_architecture"])+
            str(model_config["optimizer"])+
            str(model_config["lr"])+
            str(model_config["__loss_options__"]["siamese_loss"])+"cosine"+
            str(configurations["BATCH_SIZE"])+
            "-"+str("net")+"_{loss:.4f}.h5",
            monitor="loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )

        # Define the Keras TensorBoard callback.
        logdir = "../logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        model.optimizer.momentum = 0.9
        model.fit(
            data,
            epochs=model_config["epochs"],
            callbacks=[tensorboard_callback, checkpoint, csv_log],
            verbose=1
        )

if __name__ == "__main__":
    try:
        tf.compat.v1.app.run(main)

    except SystemExit:
        pass

# EOL

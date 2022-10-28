"""
Evaluation of the Trained Models
"""

import json
import tensorflow as tf
from create_model import get_model
from data.dataset_generator import generate_data_for_classifier, generate_data_for_siamese
from train.py import weights_dir

with open('configurations.json', encoding='utf-8') as load_value:
    configurations = json.load(load_value)

with open(
    "model_specifications.json", encoding='utf-8') as specs:
    model_config = json.load(specs)

def evaluator(model_config):
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
	    latest = tf.train.latest_checkpoint(weights_dir)
	    # Create a new model instance
		model = get_model(model_config)

		# Load the previously saved weights
		model.load_weights(latest)

		# Re-evaluate the model
		loss, acc = model.evaluate(data, verbose=2)
		print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# EOF



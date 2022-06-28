import os, sys
sys.path.append(os.getcwd())

from model.create_model import get_model
from assets.utils import json, tf
from data.dataset_generator import generate_data_for_siamese
from absl import app
from datetime import datetime, timedelta

config_file = 'configurations.json'

with open(config_file) as load_value:
    configurations = json.load(load_value)

print(configurations)
with open(configurations["MODEL_DIR"] + "/" + "model_specifications.json") as specs:
    model_config = json.load(specs)

def main(_args):

    # Loading the Data

    datax, datay = generate_data_for_siamese(configurations["DATA_DIR"], configurations["IMAGE_DIR"])


    # Loading the model

    WEIGHTS_DIR = "model/siamese/weights"
    model = get_model(model_config)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        WEIGHTS_DIR + "/" + model_config["base_architecture"] + "/siam-{epoch}-"+str(model_config["lr"])+"-"+str("net")+"_{loss:.4f}.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    # stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=cfg.TRAIN.PATIENCE, mode="min")
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.6, patience=5, min_lr=1e-6, verbose=1,
    #                                                  mode="min")

    # Defining the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


    model.fit(
        datax,datay,
        epochs=model_config["epochs"],
        callbacks=[tensorboard_callback, checkpoint],
        verbose=1
    )


if __name__ == "__main__":
    try:
        tf.compat.v1.app.run(main)

    except SystemExit:
        pass


# EOL
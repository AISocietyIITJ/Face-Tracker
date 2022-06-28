import numpy as np
import cv2
import pickle
import imutils
import json
from multiprocessing import Process
import time
import shutil
import random

import tensorflow as tf
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D,Dense,MaxPool2D,Flatten,Dropout,Add,Input,Layer,Lambda
from tensorflow.keras.activations import relu,softmax,tanh
import tensorflow.keras.losses as losses
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.applications as applications
import tensorflow_addons as tfa
import tensorflow_addons.losses as siamese_losses
import keras.backend as K

# EOL
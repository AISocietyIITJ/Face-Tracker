import os, sys
from pkgutil import get_data
sys.path.append(os.getcwd())
from assets.utils import ImageDataGenerator, tf, json
from data.helpers import Pair
from data.helpers import Pair, augment_pairs

def generate_data_for_siamese(DATA_DIR, IMAGE_DIR, isAugmented=False):

    # DATA_DIR = 'data'
    # IMAGE_DIR = f'{DATA_DIR}\\gestures'

    images = []
    labels = []

    # Map paths to images

    def decode_img(img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        # img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    
    for folder in os.listdir(IMAGE_DIR):
        for image in os.listdir(f'{IMAGE_DIR}/{folder}'):
            images.append(f'{IMAGE_DIR}/{folder}/{image}')
            labels.append(int(folder))
    
    pair_generator = Pair((images, labels))
    element_set_1, element_set_2, pair_labels =  pair_generator.get_pairs()

    # Evaluate the dataset

    element_set_1 = element_set_1.map(decode_img)
    element_set_2 = element_set_2.map(decode_img)
    pair_labels = pair_labels.map(lambda x: tf.one_hot(x, 2))

    if isAugmented:
        with open(DATA_DIR + '/' + "augmentations.json") as augmentation:
            augment_config = json.load(augmentation)
        
        element_set_1, element_set_2, pair_labels = augment_pairs(element_set_1, element_set_2, pair_labels, augment_config)

    return ([element_set_1, element_set_2], pair_labels)
    
# EOL
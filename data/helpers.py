import os, sys
sys.path.append(os.getcwd())

from assets.utils import K, ImageDataGenerator, random, np, tf

# pairs
class Pair(object):
    def __init__(self,data):
        x, y = data
        self.x, self.y = np.array(x), np.array(y)

    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def get_pairs(self):
        x, y = self.x, self.y
        print(x,y)
        pairs, labels = self.makePairs(len(np.unique(y)))
        element_1, element_2 = tf.data.Dataset.from_tensor_slices(pairs[:, 0]), tf.data.Dataset.from_tensor_slices(pairs[:, 1])
        labels = tf.data.Dataset.from_tensor_slices(labels)
        return (element_1, element_2, labels)

    def makePairs(self, num_classes):
        num_classes = num_classes
        x, y = self.x, self.y
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        for idx1 in range(len(x)):
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]
            
            labels += list([1])
            pairs += [[x1, x2]]

            label2 = random.randint(0, num_classes-1)
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]
            
            labels += list([0])
            pairs += [[x1, x2]]
        
        return np.array(pairs), np.array(labels)

class Augment(object):

    def rotate_img(img):
        img = tf.keras.layers.RandomRotation(0.2)(img)
        return img
    
    def zoom_img(img):
        img = tf.keras.layers.RandomZoom(0.5)(img)
        return img

    def shift_img(img):
        img = tf.keras.layers.RandomShift(0.5)(img)
        return img

    def flip_img(img):
        img = tf.keras.layers.RandomFlip()(img)
        return img

    def shear_img(img):
        img = tf.keras.preprocessing.image.random_shear(img, 0.2)
        return img


def augment_pairs(image_data_1,image_data_2,labels,augmentation_config):

    augmented_image_data_1 = image_data_1
    augmented_image_data_2 = image_data_2
    augmented_labels = labels

    if "rotation_range" in augmentation_config:
        rotated_1 = image_data_1.map(Augment.rotate_img)
        rotated_2 = image_data_2.map(Augment.rotate_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(rotated_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(rotated_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "width_shift_range" in augmentation_config:
        w_shifted_1 = image_data_1.map(Augment.shift_img)
        w_shifted_2 = image_data_2.map(Augment.shift_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(w_shifted_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(w_shifted_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "height_shift_range" in augmentation_config:
        h_shifted_1 = image_data_1.map(Augment.shift_img)
        h_shifted_2 = image_data_2.map(Augment.shift_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(h_shifted_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(h_shifted_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "zoom_range" in augmentation_config:
        zoomed_1 = image_data_1.map(Augment.zoom_img)
        zoomed_2 = image_data_2.map(Augment.zoom_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(zoomed_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(zoomed_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "flip_horizontal" in augmentation_config:
        flipped_1 = image_data_1.map(Augment.flip_img)
        flipped_2 = image_data_2.map(Augment.flip_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(flipped_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(flipped_2)
        augmented_labels = augmented_labels.concatenate(labels)

    if "shear_range" in augmentation_config:
        sheared_1 = image_data_1.map(Augment.shear_img)
        sheared_2 = image_data_2.map(Augment.shear_img)

        augmented_image_data_1 = augmented_image_data_1.concatenate(sheared_1)
        augmented_image_data_2 = augmented_image_data_2.concatenate(sheared_2)
        augmented_labels = augmented_labels.concatenate(labels)

    return (augmented_image_data_1,augmented_image_data_2,augmented_labels)


# EOL
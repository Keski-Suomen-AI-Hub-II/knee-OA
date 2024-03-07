import os

import numpy as np
from skimage.color import gray2rgb
from skimage.io import imread
from sklearn.utils import shuffle as sk_shuffle
import tensorflow as tf


class DataLoader:

    def __init__(self,
                 dirname1,
                 dirname2,
                 input_shape,
                 random_state=None,
                 n_classes=5,
                 input1_name='input1',
                 input2_name='input2'):
        self.dirname1 = dirname1
        self.dirname2 = dirname2
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.random_state = random_state
        self.input1_name = input1_name
        self.input2_name = input2_name

        # Output will always be RGB, i.e. 3 channels.
        self.output_shape = (self.input_shape[0], self.input_shape[1], 3)

    def filepaths(self):
        """Return a list of all image filepaths in dirname1."""
        filepaths = []
        for dirpath, _, filenames in os.walk(self.dirname1):
            for filename in filenames:
                filepath = os.sep.join([dirpath, filename])
                filepaths.append(filepath)
        if self.random_state is not None:
            filepaths = sk_shuffle(filepaths, random_state=self.random_state)
        return filepaths

    def gen_data(self):
        convert_to_rgb = len(self.input_shape) < 3
        filepaths = self.filepaths()
        for filepath1 in filepaths:
            classname = int(filepath1.split(os.sep)[-2])
            label = np.zeros(self.n_classes)
            label[classname] = 1.0
            filepath2 = filepath1.replace(self.dirname1, self.dirname2, 1)
            img1 = imread(filepath1).astype('float32')
            img2 = imread(filepath2).astype('float32')
            if convert_to_rgb:
                img1 = gray2rgb(img1)
                img2 = gray2rgb(img2)
            yield ({self.input1_name: img1, self.input2_name: img2}, label)

    def load_as_dataset(self, batch_size):
        ds = tf.data.Dataset.from_generator(
            self.gen_data,
            output_signature=({
                self.input1_name:
                tf.TensorSpec(shape=self.output_shape,
                              dtype=tf.float32,
                              name=None),
                self.input2_name:
                tf.TensorSpec(shape=self.output_shape,
                              dtype=tf.float32,
                              name=None)
            },
                              tf.TensorSpec(shape=(self.n_classes),
                                            dtype=tf.float32,
                                            name=None))).batch(batch_size)
        return ds

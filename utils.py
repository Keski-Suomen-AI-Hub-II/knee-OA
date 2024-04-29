#!/usr/bin/env python3

import os
import random as rand
import shutil

import numpy as np
import skimage as ski
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.config import list_physical_devices, set_visible_devices
from tensorflow.config.experimental import set_memory_growth


def reserve_gpu(id):
    if id < 0:
        return
    gpus = list_physical_devices('GPU')
    if gpus:
        gpu = gpus[id]
        set_visible_devices(gpu, 'GPU')
        set_memory_growth(gpu, True)


def write_confusion_matrix(model, data, filepath, desc_text):
    """Save confusion matrix to a given filepath."""
    labels = np.concatenate([label for _, label in data], axis=0)
    labels = tf.math.argmax(labels, axis=-1)
    preds = model.predict(data)
    preds = tf.math.argmax(preds, axis=-1)
    cm = confusion_matrix(labels, preds)
    with open(filepath, mode='a') as f:
        f.write(desc_text)
        f.write(str(cm))
        f.write('\n')
        for i in range(cm.shape[0]):
            class_acc = cm[i, i] / sum(cm[i])
            f.write('Class {}: {:.6f}\n'.format(i, class_acc))
        f.write('\n')


def enhance_contrast(src_path, dst_path):
    """Perform histogram equalization on all the images inside src_path."""
    for src_dirpath, _, filenames in os.walk(src_path):
        dst_dirpath = src_dirpath.replace(src_path, dst_path, 1)
        os.makedirs(dst_dirpath, exist_ok=True)
        for filename in filenames:
            src_filepath = os.path.sep.join([src_dirpath, filename])
            dst_filepath = os.path.sep.join([dst_dirpath, filename])
            img = ski.io.imread(src_filepath)
            img_enh = ski.exposure.equalize_hist(img)
            img_enh = ski.util.img_as_ubyte(img_enh)  # Convert to 8-bit ints.
            ski.io.imsave(dst_filepath, img_enh)


def resize_imgs(src_path, dst_path, shape_to_save):
    """Save resized images in dst_path."""
    for src_dirpath, _, filenames in os.walk(src_path):
        dst_dirpath = src_dirpath.replace(src_path, dst_path, 1)
        os.makedirs(dst_dirpath, exist_ok=True)
        for filename in filenames:
            src_filepath = os.path.sep.join([src_dirpath, filename])
            dst_filepath = os.path.sep.join([dst_dirpath, filename])
            img = ski.io.imread(src_filepath)
            img_resized = ski.transform.resize(img,
                                               shape_to_save,
                                               preserve_range=True)
            img_resized = img_resized.astype('uint8')
            ski.io.imsave(dst_filepath, img_resized)


def copy_random_files(src_path, dst_path, n_files):
    """Copy random files to dst_path."""
    all_fnames = os.listdir(src_path)
    rand_fnames = rand.choices(all_fnames, k=n_files)
    os.makedirs(dst_path, exist_ok=True)
    for fname in rand_fnames:
        fpath = os.path.sep.join([src_path, fname])
        shutil.copy(fpath, dst_path)


def main():
    templates_path = '../eminentia/eminentia_samples/output'
    datapath = '../eminentia/dataset_i/data'
    cropped_datapath = '../eminentia/autocropped_from_hist_eq_data'
    cropped_datapath_enh = '../eminentia/dataset_i/autocropped'
    shape_to_save = (224, 224)
    crop_images(datapath, cropped_datapath, templates_path, shape_to_save)
    enhance_contrast(cropped_datapath, cropped_datapath_enh)


if __name__ == '__main__':
    main()

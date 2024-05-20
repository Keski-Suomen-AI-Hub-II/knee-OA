#!/usr/bin/env python3

import os
import random as rand
import shutil

import numpy as np
import skimage as ski
import tensorflow as tf
from sklearn import metrics
from tensorflow.config import list_physical_devices, set_visible_devices
from tensorflow.config.experimental import set_memory_growth
from tensorflow.keras import callbacks


def reserve_gpu(id):
    if id < 0:
        return
    gpus = list_physical_devices('GPU')
    if gpus:
        gpu = gpus[id]
        set_visible_devices(gpu, 'GPU')
        set_memory_growth(gpu, True)


def convert_from_1hot(model, data):
    """Convert labels and preds from one-hot encoding."""
    labels = np.concatenate([label for _, label in data], axis=0)
    labels = tf.math.argmax(labels, axis=-1)
    preds = model.predict(data)
    preds = tf.math.argmax(preds, axis=-1)
    return labels, preds


def write_metrics(model, data, filepath):
    """Write classification metrics to a given filepath."""
    # Get the labels and preds.
    labels, preds = convert_from_1hot(model, data)

    # Calculate and save the metrics.
    report = metrics.classification_report(labels, preds)
    with open(filepath, mode='a') as f:
        f.write(report)


def write_confusion_matrix(model, data, filepath, desc_text):
    """Save confusion matrix to a given filepath."""
    # Get the labels and preds.
    labels, preds = convert_from_1hot(model, data)

    # Calculate and save the confusion matrix.
    cm = metrics.confusion_matrix(labels, preds)
    with open(filepath, mode='a') as f:
        f.write(desc_text)
        f.write(str(cm))
        f.write('\n')
        for i in range(cm.shape[0]):
            class_acc = cm[i, i] / sum(cm[i])
            f.write('Class {}: {:.6f}\n'.format(i, class_acc))
        f.write('\n')


def visualize_confusion_matrix(model, data, filepath):
    """Save visual confusion matrix to a given filepath."""
    labels, preds = convert_from_1hot(model, data)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(labels, preds)
    disp.plot().figure_.savefig(filepath)


def flip_images(src_path, dst_path, substr='L'):
    """Flip the images with the given substring in name."""
    for src_dirpath, _, filenames in os.walk(src_path):
        dst_dirpath = src_dirpath.replace(src_path, dst_path, 1)
        os.makedirs(dst_dirpath, exist_ok=True)
        for filename in filenames:
            if filename.count(substr) > 0:
                src_filepath = os.path.sep.join([src_dirpath, filename])
                dst_filepath = os.path.sep.join([dst_dirpath, filename])
                img = ski.io.imread(src_filepath)
                flipped = np.fliplr(img)
                ski.io.imsave(dst_filepath, flipped)


def enhance_contrast(src_path, dst_path):
    """Perform histogram equalization on all the images inside src_path."""
    for src_dirpath, _, filenames in os.walk(src_path):
        dst_dirpath = src_dirpath.replace(src_path, dst_path, 1)
        os.makedirs(dst_dirpath, exist_ok=True)
        for filename in filenames:
            src_filepath = os.path.sep.join([src_dirpath, filename])
            dst_filepath = os.path.sep.join([dst_dirpath, filename])
            img = ski.io.imread(src_filepath)
            # Only one channel.
            if len(img) > 2:
                img = img[:, :, 0]
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


def replace_from_filenames(dirpath, old, new):
    """Replace specific substring in all the files inside directory."""
    for fname in os.listdir(dirpath):
        fpath = os.path.sep.join([dirpath, fname])
        fname_new = fname.replace(old, new)
        fpath_new = os.path.sep.join([dirpath, fname_new])
        os.rename(fpath, fpath_new)


def train_model(model, ds_train, ds_val, epochs, trainlog_path,
                checkpoint_dirpath):
    """Train and write trainlog."""
    checkpoint_path = os.path.sep.join([checkpoint_dirpath, 'checkpoint'])
    cbs_list = [
        callbacks.CSVLogger(trainlog_path, append=True),
        callbacks.EarlyStopping(monitor='val_accuracy',
                                patience=10,
                                verbose=0,
                                mode='max'),
        callbacks.ModelCheckpoint(checkpoint_path,
                                  monitor='val_accuracy',
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='max')
    ]
    model.fit(x=ds_train,
              validation_data=ds_val,
              epochs=epochs,
              verbose=0,
              callbacks=cbs_list)
    model.load_weights(checkpoint_path)
    shutil.rmtree(checkpoint_dirpath, ignore_errors=True)


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

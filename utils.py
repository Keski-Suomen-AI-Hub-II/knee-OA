#!/usr/bin/env python3

import os
import random as rand
import shutil

import numpy as np
import skimage as ski
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import KFold
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


def write_metrics(model, data, filepath, desc_text=''):
    """Write classification metrics to a given filepath."""
    # Get the labels and preds.
    labels, preds = convert_from_1hot(model, data)
    # Get the confusion matrix.
    cm = metrics.confusion_matrix(labels, preds)

    # Calculate and save the metrics.
    report = metrics.classification_report(labels, preds)
    with open(filepath, mode='a') as f:
        if desc_text:
            f.write(desc_text)
        f.write(str(cm))
        f.write(report)


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


def crop_images(src_path, dst_path, x_min, x_max, y_min, y_max):
    """Save the cropped images."""
    for src_dirpath, _, filenames in os.walk(src_path):
        dst_dirpath = src_dirpath.replace(src_path, dst_path, 1)
        os.makedirs(dst_dirpath, exist_ok=True)
        for filename in filenames:
            src_filepath = os.path.sep.join([src_dirpath, filename])
            dst_filepath = os.path.sep.join([dst_dirpath, filename])
            img = ski.io.imread(src_filepath)
            # Only one channel.
            if len(img) > 2:
                img = img[:, :]
            img_cropped = img[y_min:y_max, x_min:x_max]
            ski.io.imsave(dst_filepath, img_cropped)


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
            if len(img.shape) > 2:
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


def copy_move_files(src_path, dst_path, filenames, move=False):
    """Copy or move files with listed filenames to dst_path."""
    os.makedirs(dst_path, exist_ok=True)
    for fname in filenames:
        fpath = os.path.sep.join([src_path, fname])
        if move:
            shutil.move(fpath, dst_path)
        else:
            shutil.copy(fpath, dst_path)


def copy_move_random_files(src_path, dst_path, n_files, move=False):
    """Copy or move random files to dst_path."""
    all_fnames = os.listdir(src_path)
    rand_fnames = rand.sample(all_fnames, n_files)
    copy_move_files(src_path, dst_path, rand_fnames, move=move)


def move_random_files(src_path, dst_path, n_files):
    """Move random files to dst_path."""
    copy_move_random_files(src_path, dst_path, n_files, move=True)


def replace_from_filenames(src_path, old, new):
    """Replace specific substring in all the files inside directory."""
    for src_dirpath, _, filenames in os.walk(src_path):
        for fname in filenames:
            fpath = os.path.sep.join([src_dirpath, fname])
            fname_new = fname.replace(old, new)
            fpath_new = os.path.sep.join([src_dirpath, fname_new])
            os.rename(fpath, fpath_new)


def split_files(src_root1,
                src_root2,
                dst_root,
                classname,
                k=5,
                dir1_specifier='data',
                dir2_specifier='cropped'):
    """Split files of one class for k-fold cross validation."""
    # Source directories are assumed to be under the source root directories.
    classname = str(classname)
    src_dir1 = os.path.sep.join([src_root1, classname])
    src_dir2 = os.path.sep.join([src_root2, classname])

    kf = KFold(n_splits=k, shuffle=True)
    filenames = os.listdir(src_dir1)

    # Iterate over the splits.
    split_ind = 0
    for train_inds, val_inds in kf.split(filenames):
        # Get the filenames for training and validation.
        train_names = [filenames[i] for i in train_inds]
        val_names = [filenames[i] for i in val_inds]

        # Make directories for training and validation data.
        split_name = 'split{}'.format(split_ind)
        dst_traindir1 = os.path.sep.join(
            [dst_root, split_name, dir1_specifier, 'train', classname])
        dst_traindir2 = os.path.sep.join(
            [dst_root, split_name, dir2_specifier, 'train', classname])
        os.makedirs(dst_traindir1, exist_ok=True)
        os.makedirs(dst_traindir2, exist_ok=True)
        dst_valdir1 = os.path.sep.join(
            [dst_root, split_name, dir1_specifier, 'val', classname])
        dst_valdir2 = os.path.sep.join(
            [dst_root, split_name, dir2_specifier, 'val', classname])
        os.makedirs(dst_valdir1, exist_ok=True)
        os.makedirs(dst_valdir2, exist_ok=True)

        # Copy files to destination directories.
        copy_move_files(src_dir1, dst_traindir1, train_names)
        copy_move_files(src_dir2, dst_traindir2, train_names)
        copy_move_files(src_dir1, dst_valdir1, val_names)
        copy_move_files(src_dir2, dst_valdir2, val_names)

        split_ind += 1


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
    pass


if __name__ == '__main__':
    main()

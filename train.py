#!/usr/bin/env python3

import os
import shutil
import sys

from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid

from parallel_network import ParallelNetwork
import utils


def train_model(model, ds_train, ds_val, config, trainlog_path,
                checkpoint_dirpath):
    checkpoint_path = os.path.sep.join([checkpoint_dirpath, 'checkpoint'])
    cbs_list = [
        CSVLogger(trainlog_path, append=True),
        ModelCheckpoint(checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True,
                        mode='min')
    ]
    history = model.fit(x=ds_train,
                        validation_data=ds_val,
                        epochs=config['n_epochs'],
                        verbose=0,
                        callbacks=cbs_list)
    model.load_weights(checkpoint_path)
    shutil.rmtree(checkpoint_dirpath, ignore_errors=True)
    return history


def write_confusion_matrix(model, data, filepath, desc_text):
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


def grid_search(configs, search_path, src_shape, dest_shape):
    trainlog_path = os.path.sep.join([search_path, 'training.log'])
    checkpoint_dirpath = os.path.sep.join([search_path, 'temp'])

    # Iterate over the configurations.
    for i, config in configs:
        with open(trainlog_path, mode='a') as f:
            f.write('Configuration {}: {}\n'.format(i, config))
        network = ParallelNetwork(dest_shape,
                                  config['base_models'],
                                  weights=config['weights'],
                                  dropout=config['dropout'])
        model = network.build()
        model.compile(optimizer=Adam(learning_rate=config['lr'], beta_1=.85),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # To save GPU's memory, explicitly use CPU to load the data.
        with tf.device('CPU'):
            ds_train = utils.load_as_dataset('data/train',
                                             'data_autocropped/train',
                                             src_shape,
                                             config['batch_size'],
                                             random_seed=313)
            ds_val = utils.load_as_dataset('data/val', 'data_autocropped/val',
                                           src_shape, config['batch_size'])
            history = train_model(model, ds_train, ds_val, config,
                                  trainlog_path, checkpoint_dirpath)
        for (data, text) in [(ds_train, 'Training data:\n'),
                             (ds_val, 'Validation data:\n')]:
            write_confusion_matrix(model, data, trainlog_path, text)


def main():
    # Different configurations consist of several parameter combinations.
    param_grid = {
        'lr': [1e-5],  #[1e-4, 5e-5, 1e-5],
        'n_epochs': [25],
        'base_models': [('vgg-19', 'vgg-19'), ('inception_v3', 'inception_v3'),
                        ('xception', 'xception')],
        'batch_size': [16],
        'weights': ['imagenet'],
        'dropout': [0, .1, .2, .3]
    }
    configs = enumerate(list(ParameterGrid(param_grid)))

    # Grid search directory is named using starting time.
    time = datetime.now().strftime('%y-%m-%d-%H%M%S')
    search_path = 'grid_search_{}'.format(time)
    if not os.path.exists(search_path):
        os.mkdir(search_path)

    # # Perform grid search.
    gpu_id = int(sys.argv[1])
    utils.reserve_gpu(gpu_id)
    src_shape = (224, 224)
    dest_shape = (224, 224, 3)
    grid_search(configs, search_path, src_shape, dest_shape)


if __name__ == '__main__':
    main()

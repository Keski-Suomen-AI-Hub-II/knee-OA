#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam

from data_loader import DataLoader
from parallel_network import ParallelNetwork
import utils


def train_model(model, ds_train, ds_val, epochs, trainlog_path,
                checkpoint_dirpath):
    checkpoint_path = os.path.sep.join([checkpoint_dirpath, 'checkpoint'])
    cbs_list = [
        callbacks.CSVLogger(trainlog_path, append=True),
        callbacks.EarlyStopping(monitor='val_loss',
                                patience=10,
                                verbose=0,
                                mode='min'),
        callbacks.ModelCheckpoint(checkpoint_path,
                                  monitor='val_loss',
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='min')
    ]
    model.fit(x=ds_train,
              validation_data=ds_val,
              epochs=epochs,
              verbose=0,
              callbacks=cbs_list)
    model.load_weights(checkpoint_path)
    shutil.rmtree(checkpoint_dirpath, ignore_errors=True)


def grid_search(dir1_train, dir2_train, dir1_val, dir2_val, classes, configs,
                training_path, src_shape, dest_shape, n_epochs, batch_size,
                n_save):
    best_results = []
    trainlog_path = os.path.sep.join([training_path, 'training.log'])
    checkpoint_dirpath = os.path.sep.join([training_path, 'temp'])

    # Initialize data loader for training and evaluation data.
    input_names = ('input1', 'input2')
    dl_train = DataLoader(dir1_train,
                          dir2_train,
                          src_shape,
                          input_names,
                          n_classes=classes,
                          random_state=17)
    dl_val = DataLoader(dir1_val,
                        dir2_val,
                        src_shape,
                        input_names,
                        n_classes=classes)
    # Get the datasets.
    ds_train = dl_train.load_as_dataset(batch_size)
    ds_val = dl_val.load_as_dataset(batch_size)

    # Iterate over the configurations.
    branch_names = ('branch1', 'branch2')
    for i, config in configs:
        with open(trainlog_path, mode='a') as f:
            f.write('Configuration {}: {}\n'.format(i, config))
        tf.keras.backend.clear_session()
        # Build the model, freeze its branches' weights, and
        # compile.
        network = ParallelNetwork(dest_shape,
                                  config['base_models'],
                                  branch_names,
                                  input_names,
                                  classes=classes,
                                  weights=config['weights'],
                                  dropout=config['dropout'])
        model = network.build()
        for branch_name in branch_names:
            branch = model.get_layer(branch_name)
            branch.trainable = False
        if classes == 2:
            model.compile(optimizer=Adam(learning_rate=config['lr']),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            model.compile(optimizer=Adam(learning_rate=config['lr']),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        # Train the model. With the best weights, print confusion matrix for
        # the training and validation data.
        train_model(model, ds_train, ds_val, n_epochs, trainlog_path,
                    checkpoint_dirpath)
        for (data, text) in [(ds_train, 'Training data:\n'),
                             (ds_val, 'Validation data:\n')]:
            utils.write_confusion_matrix(model, data, trainlog_path, text)

        # If models are to be saved, then:
        #   calculate val accuracy
        #   save the model
        #   make sure that only the n_save best models are saved.
        if n_save > 0:
            _, val_accuracy = model.evaluate(ds_val)
            best_results.append((i, val_accuracy, model))
            if len(best_results) > n_save:
                best_results.sort(key=lambda el: el[1], reverse=True)
                del best_results[-1]
    return best_results


def save_models(dirpath, results):
    for i, metric, model in results:
        filename = '{}_{:.3f}.h5'.format(i, metric)
        filepath = os.path.sep.join([dirpath, filename])
        model.trainable = True
        model.save_weights(filepath)


def main():
    # Get the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('branch1',
                        help='branch 1 convolutional base',
                        type=str)
    parser.add_argument('branch2',
                        help='branch 2 convolutional base',
                        type=str)
    parser.add_argument('dir1_train',
                        help='directory 1 of training data',
                        type=str)
    parser.add_argument('dir2_train',
                        help='directory 2 of training data',
                        type=str)
    parser.add_argument('dir1_val',
                        help='directory 1 of validation data',
                        type=str)
    parser.add_argument('dir2_val',
                        help='directory 2 of validation data',
                        type=str)

    parser.add_argument('--gpu_id', help='id of GPU', type=int, default=0)
    parser.add_argument('--n_save',
                        help='how many models to save',
                        type=int,
                        default=1)
    parser.add_argument('--classes',
                        help='number of classes',
                        type=int,
                        default=5)
    parser.add_argument('--weights1',
                        help='weights of branch 1',
                        type=str,
                        default='imagenet')
    parser.add_argument('--weights2',
                        help='weights of branch2',
                        type=str,
                        default='imagenet')
    parser.add_argument('--bsize', help='batch size', type=int, default=8)
    parser.add_argument('--epochs',
                        help='maximum number of epochs',
                        type=int,
                        default=100)
    args = parser.parse_args()

    # Different configurations consist of several parameter combinations.
    param_grid = {
        'lr': [1e-4, 1e-5],  #[1e-4, 5e-5, 1e-5],
        'base_models': [(args.branch1, args.branch2)],
        'weights': [(args.weights1, args.weights2)],
        'dropout': [0]  #[0, .1, .2, .3]
    }
    configs = enumerate(list(ParameterGrid(param_grid)))

    # Name grid search directory by starting time.
    time = datetime.now().strftime('%y-%m-%d-%H%M%S')
    training_path = 'training_{}'.format(time)
    if not os.path.exists(training_path):
        os.mkdir(training_path)

    # Perform grid search and save the best models.
    utils.reserve_gpu(args.gpu_id)
    src_shape = (224, 224)
    dest_shape = (224, 224, 3)
    best_results = grid_search(args.dir1_train, args.dir2_train, args.dir1_val,
                               args.dir2_val, args.classes, configs,
                               training_path, src_shape, dest_shape,
                               args.epochs, args.bsize, args.n_save)
    if args.n_save > 0:
        save_models(training_path, best_results)


if __name__ == '__main__':
    main()

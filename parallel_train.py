#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.optimizers import Adam

import utils
from data_loader import DataLoader
from parallel_network import ParallelNetwork


def grid_search(configs, classes, traindata_dirs, valdata_dirs, training_path,
                src_shape, dest_shape, n_epochs, batch_size, n_save):
    best_results = []
    trainlog_path = os.path.sep.join([training_path, 'training.log'])
    checkpoint_dirpath = os.path.sep.join([training_path, 'temp'])

    # Initialize data loader for training and evaluation data.
    dl_train = DataLoader(traindata_dirs[0],
                          traindata_dirs[1],
                          src_shape,
                          n_classes=classes,
                          random_state=17)
    dl_val = DataLoader(valdata_dirs[0],
                        valdata_dirs[1],
                        src_shape,
                        n_classes=classes)
    # Get the datasets.
    ds_train = dl_train.load_as_dataset(batch_size)
    ds_val = dl_val.load_as_dataset(batch_size)

    # Iterate over the configurations.
    for i, config in configs:
        with open(trainlog_path, mode='a') as f:
            f.write('Configuration {}: {}\n'.format(i, config))
        tf.keras.backend.clear_session()
        # Build and compile the model.
        network = ParallelNetwork(dest_shape,
                                  config['base_model'],
                                  classes=classes,
                                  weights=config['weights'],
                                  dropout=config['dropout'])
        model = network.build()
        model.compile(optimizer=Adam(learning_rate=config['lr']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model. With the best weights, print confusion matrix for
        # the training and validation data.
        utils.train_model(model, ds_train, ds_val, n_epochs, trainlog_path,
                          checkpoint_dirpath)
        for (data, text) in [(ds_train, 'Training data:\n'),
                             (ds_val, 'Validation data:\n')]:
            utils.write_confusion_matrix(model, data, trainlog_path, text)

        # If models are to be saved, then:
        #   calculate val accuracy
        #   save the model
        #   make sure that only the n_save best models are saved.
        if n_save > 0:
            _, val_acc = model.evaluate(ds_val)
            best_results.append((i, val_acc, model))
            if len(best_results) > n_save:
                best_results.sort(key=lambda el: el[1], reverse=True)
                del best_results[-1]
    return best_results


def save_models(dirpath, results):
    for i, metric, model in results:
        filename = 'config_{}_val_acc_{:.3f}.h5'.format(i, metric)
        filepath = os.path.sep.join([dirpath, filename])
        model.save_weights(filepath)


def main():
    # Get the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('base_model',
                        help='convolutional base model',
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
    parser.add_argument('--weights',
                        help='weights of the convolutional base',
                        type=str,
                        default='imagenet')
    parser.add_argument('--bsize', help='batch size', type=int, default=8)
    parser.add_argument('--epochs',
                        help='maximum number of epochs',
                        type=int,
                        default=200)
    args = parser.parse_args()

    # Different configurations consist of several parameter combinations.
    param_grid = {
        'base_model': [args.base_model],
        'weights': [args.weights],
        'lr': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'dropout': [0, .1, .2, .3]
    }
    configs = enumerate(list(ParameterGrid(param_grid)))

    # Name grid search directory by branches and starting time.
    time = datetime.now().strftime('%y-%m-%d-%H%M%S')
    training_path = 'parallel-{}-class_{}_{}'.format(args.classes,
                                                     args.base_model, time)
    if not os.path.exists(training_path):
        os.mkdir(training_path)

    # Perform grid search and save the best models.
    traindata_dirs = (args.dir1_train, args.dir2_train)
    valdata_dirs = (args.dir1_val, args.dir2_val)
    src_shape = (224, 224)
    dest_shape = (224, 224, 3)
    utils.reserve_gpu(args.gpu_id)
    best_results = grid_search(configs, args.classes, traindata_dirs,
                               valdata_dirs, training_path, src_shape,
                               dest_shape, args.epochs, args.bsize,
                               args.n_save)
    if args.n_save > 0:
        save_models(training_path, best_results)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import os
from datetime import datetime

import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory

import utils
from single_network import SingleNetwork


def grid_search(configs, classes, traindata_dir, valdata_dir, training_path,
                src_shape, dest_shape, n_epochs, batch_size, n_save):
    best_results = []
    trainlog_path = os.path.sep.join([training_path, 'training.log'])
    checkpoint_dirpath = os.path.sep.join([training_path, 'temp'])

    # Get the datasets.
    if len(dest_shape) < 3:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'
    ds_train = image_dataset_from_directory(traindata_dir,
                                            label_mode='categorical',
                                            color_mode=color_mode,
                                            batch_size=batch_size,
                                            image_size=(dest_shape[0],
                                                        dest_shape[1]),
                                            shuffle=True)
    ds_val = image_dataset_from_directory(valdata_dir,
                                          label_mode='categorical',
                                          color_mode=color_mode,
                                          batch_size=batch_size,
                                          image_size=(dest_shape[0],
                                                      dest_shape[1]),
                                          shuffle=False)

    # Iterate over the configurations.
    for i, config in configs:
        with open(trainlog_path, mode='a') as f:
            f.write('Configuration {}: {}\n'.format(i, config))
        tf.keras.backend.clear_session()
        # Build and compile the model.
        network = SingleNetwork(dest_shape,
                                config['base_model'],
                                classes=classes,
                                weights=config['weights'],
                                dropout=config['dropout'])
        model = network.build()
        model.compile(optimizer=Adam(learning_rate=config['lr']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model. With the best weights, print metrics for
        # the validation data.
        utils.train_model(model, ds_train, ds_val, n_epochs, trainlog_path,
                          checkpoint_dirpath)
        utils.write_metrics(model, ds_val, trainlog_path, 'Validation data:\n')

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
    parser.add_argument('dir_train', help='training data directory', type=str)
    parser.add_argument('dir_val', help='validation data directory', type=str)

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
    parser.add_argument('--bsize', help='batch size', type=int, default=16)
    parser.add_argument('--epochs',
                        help='maximum number of epochs',
                        type=int,
                        default=200)
    args = parser.parse_args()

    # Different configurations consist of several parameter combinations.
    param_grid = {
        'base_model': [args.base_model],
        'weights': [args.weights],
        'lr': [1e-4, 1e-5, 1e-6],
        'dropout': [0, .3]
    }
    configs = enumerate(list(ParameterGrid(param_grid)))

    # Name grid search directory by convnet and starting time.
    time = datetime.now().strftime('%y-%m-%d-%H%M%S')
    training_path = 'single-{}-class_{}_{}'.format(args.classes,
                                                   args.base_model, time)
    if not os.path.exists(training_path):
        os.mkdir(training_path)

    # Perform grid search and save the best models.
    traindata_dir = args.dir_train
    valdata_dir = args.dir_val
    src_shape = (100, 224)
    dest_shape = (100, 224, 3)
    utils.reserve_gpu(args.gpu_id)
    best_results = grid_search(configs, args.classes, traindata_dir,
                               valdata_dir, training_path, src_shape,
                               dest_shape, args.epochs, args.bsize,
                               args.n_save)
    if args.n_save > 0:
        save_models(training_path, best_results)


if __name__ == '__main__':
    main()

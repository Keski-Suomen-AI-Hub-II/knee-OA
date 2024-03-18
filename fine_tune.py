#!/usr/bin/env python3

import argparse
import os
import shutil
from datetime import datetime

from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam

from data_loader import DataLoader
from parallel_network import ParallelNetwork
import utils


def train_model(model, ds_train, ds_val, epochs, learning_rate, trainlog_path,
                checkpoint_dirpath):
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


def build_and_fine_tune(branch1, branch2, lr, dir1_train, dir2_train, dir1_val,
                        dir2_val, classes, tuning_path, src_shape, dest_shape,
                        n_epochs, batch_size, dropout_rate, weights):
    trainlog_path = os.path.sep.join([tuning_path, 'training.log'])
    checkpoint_dirpath = os.path.sep.join([tuning_path, 'temp'])
    input_names = ('input1', 'input2')
    branch_names = ('branch1', 'branch2')

    # Get the data.
    dl_train = DataLoader(dir1_train,
                          dir2_train,
                          src_shape,
                          input_names,
                          n_classes=classes,
                          random_state=381)
    dl_val = DataLoader(dir1_val,
                        dir2_val,
                        src_shape,
                        input_names,
                        n_classes=classes)
    ds_train = dl_train.load_as_dataset(batch_size)
    ds_val = dl_val.load_as_dataset(batch_size)

    # Build and compile the model.
    network = ParallelNetwork(dest_shape, (branch1, branch2),
                              branch_names,
                              input_names,
                              classes=classes,
                              dropout=dropout_rate)
    model = network.build()
    model.load_weights(weights)
    if classes == 2:
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # Train the model. With the best weights, print confusion matrix for the
    # training and validation data.
    train_model(model, ds_train, ds_val, n_epochs, lr, trainlog_path,
                checkpoint_dirpath)
    for (data, text) in [(ds_train, 'Training data:\n'),
                         (ds_val, 'Validation data:\n')]:
        utils.write_confusion_matrix(model, data, trainlog_path, text)

    # Return validation accuracy and model.
    _, val_accuracy = model.evaluate(ds_val)
    return val_accuracy, model


def save_model(dirpath, metric, model):
    filename = '{:.3f}.h5'.format(metric)
    filepath = os.path.sep.join([dirpath, filename])
    model.save_weights(filepath)


def main():
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
    parser.add_argument('weights', help='path to weights', type=str)
    parser.add_argument('dropout', help='dropout rate', type=float)

    parser.add_argument('--gpu_id', help='id of GPU', type=int, default=0)
    parser.add_argument('--classes',
                        help='number of classes',
                        type=int,
                        default=5)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-6)
    parser.add_argument('--epochs',
                        help='maximum number of epochs',
                        type=int,
                        default=100)
    parser.add_argument('--bsize', help='batch size', type=int, default=8)
    args = parser.parse_args()

    # Name fine tuning directory by starting time.
    time = datetime.now().strftime('%y-%m-%d-%H%M%S')
    tuning_path = 'fine_tuning_{}'.format(time)
    if not os.path.exists(tuning_path):
        os.mkdir(tuning_path)

    utils.reserve_gpu(args.gpu_id)
    src_shape = (224, 224)
    dest_shape = (224, 224, 3)
    val_acc, model = build_and_fine_tune(args.branch1, args.branch2, args.lr,
                                         args.dir1_train, args.dir2_train,
                                         args.dir1_val, args.dir2_val,
                                         args.classes, tuning_path, src_shape,
                                         dest_shape, args.epochs, args.bsize,
                                         args.dropout, args.weights)

    save_model(tuning_path, val_acc, model)


if __name__ == '__main__':
    main()

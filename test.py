#!/usr/bin/env python3

import argparse
import os

import utils
from data_loader import DataLoader
from parallel_network import ParallelNetwork


def test_model(config, weights_path, testdata_dirs, src_shape, dest_shape,
               batch_size):
    input_names = ('input1', 'input2')
    branch_names = ('branch1', 'branch2')

    # Get the data.
    dl_test = DataLoader(testdata_dirs[0],
                         testdata_dirs[1],
                         src_shape,
                         input_names,
                         n_classes=config['classes'])
    ds_test = dl_test.load_as_dataset(batch_size)

    # Get the model.
    network = ParallelNetwork(dest_shape, config['base_models'], branch_names,
                              input_names, config['classes'])
    model = network.build()
    model.load_weights(weights_path)
    if config['classes'] == 2:
        model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # Test.
    model.evaluate(ds_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('branch1',
                        help='branch 1 convolutional base',
                        type=str)
    parser.add_argument('branch2',
                        help='branch 2 convolutional base',
                        type=str)
    parser.add_argument('dir1_test', help='directory 1 of test data', type=str)
    parser.add_argument('dir2_test', help='directory 2 of test data', type=str)
    parser.add_argument('weights', help='path to weights', type=str)

    parser.add_argument('--gpu_id', help='id of GPU', type=int, default=0)
    parser.add_argument('--classes',
                        help='number of classes',
                        type=int,
                        default=5)
    parser.add_argument('--bsize', help='batch size', type=int, default=8)
    args = parser.parse_args()

    # Define model configuration. Dropout is not needed, because the model is
    # only used in inference mode.
    model_config = {
        'classes': args.classes,
        'base_models': (args.branch1, args.branch2),
    }

    testdata_dirs = (args.dir1_test, args.dir2_test)
    src_shape = (224, 224)
    dst_shape = (224, 224, 3)
    utils.reserve_gpu(args.gpu_id)
    test_model(model_config, args.weights, testdata_dirs, src_shape, dst_shape,
               args.bsize)


if __name__ == '__main__':
    main()

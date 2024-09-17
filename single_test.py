#!/usr/bin/env python3

import os

from tensorflow.keras.preprocessing import image_dataset_from_directory

import utils
from single_network import SingleNetwork


def test_model(base_model, weights_path, classes, dir_test, dst_shape,
               batch_size, report_dir):
    # Get the data.
    if len(dst_shape) < 3:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'
    ds_test = image_dataset_from_directory(dir_test,
                                           label_mode='categorical',
                                           color_mode=color_mode,
                                           batch_size=batch_size,
                                           image_size=(dst_shape[0],
                                                       dst_shape[1]),
                                           shuffle=False)

    # Get the model.
    network = SingleNetwork(dst_shape, base_model, classes=classes)
    model = network.build()
    model.load_weights(weights_path)
    model.compile()

    # Test and save the results.
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)
    metrics_path = os.path.join(report_dir, 'metrics.txt')
    cm_path = os.path.join(report_dir, 'conf_matrix.png')
    utils.write_metrics(model, ds_test, metrics_path, desc_text='Test data:\n')
    utils.visualize_confusion_matrix(model, ds_test, cm_path)


def main():
    base_model = ''
    dir_test = ''
    weights_path = ''
    classes = 5
    batch_size = 16
    report_dir = ''

    dst_shape = (100, 224, 3)
    test_model(base_model, weights_path, classes, dir_test, dst_shape,
               batch_size, report_dir)


if __name__ == '__main__':
    main()

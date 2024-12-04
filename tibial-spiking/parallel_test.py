#!/usr/bin/env python3

import os

import utils
from data_loader import DataLoader
from parallel_network import ParallelNetwork


def test_model(base_model, weights_path, classes, dir1_test, dir2_test,
               src_shape, dst_shape, batch_size, report_dir):
    # Get the data.
    dl_test = DataLoader(dir1_test, dir2_test, src_shape, n_classes=classes)
    ds_test = dl_test.load_as_dataset(batch_size)

    # Get the model.
    network = ParallelNetwork(dst_shape, base_model, classes=classes)
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
    dir1_test = ''
    dir2_test = ''
    weights_path = ''
    classes = 5
    batch_size = 8
    report_dir = ''

    src_shape = (100, 224)
    dst_shape = (100, 224, 3)
    test_model(base_model, weights_path, classes, dir1_test, dir2_test,
               src_shape, dst_shape, batch_size, report_dir)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import copy
import datetime
import os

import sklearn as skl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import networks
import utils


def train_model(model,
                train_loader,
                val_loader,
                device,
                criterion,
                optimizer,
                trainlog_path,
                is_binary,
                n_epochs=25,
                patience=5):
    best_acc = 0.0
    best_weights = None
    patience_left = patience
    for epoch in range(n_epochs):
        model.train()
        for data in train_loader:
            # Get the inputs and labels and move them to the device.
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Forward, backward, and optimize.
            outputs = model(inputs)
            if is_binary:
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Write the metrics.
        train_preds, train_labels = utils.predictions_and_labels(
            model, train_loader, is_binary, device)
        train_acc = utils.accuracy(train_preds, train_labels)
        val_preds, val_labels = utils.predictions_and_labels(
            model, val_loader, is_binary, device)
        val_acc = utils.accuracy(val_preds, val_labels)
        with open(trainlog_path, mode='a') as f:
            f.write(f'Epoch {epoch + 1}/{n_epochs}, '
                    f'train acc: {train_acc:.5f}, '
                    f'val acc: {val_acc:.5f}\n')

        # Early stopping.
        if best_acc < val_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # Linebreak before the next configuration.
    with open(trainlog_path, mode='a') as f:
        f.write('\n')

    # Load the best weights.
    model.load_state_dict(best_weights)


def grid_search(cfgs,
                train_dir,
                val_dir,
                training_path,
                n_classes,
                device,
                save_weights=True):
    trainlog_path = os.path.join(training_path, 'training.log')
    list_configs = os.path.join(training_path, 'configs')
    list_results = os.path.join(training_path, 'results')
    is_binary = n_classes == 2

    # Iterate through the configurations.
    for i, cfg in cfgs:
        with open(trainlog_path, mode='a') as f:
            f.write(f'Configuration {i}: {cfg}\n')
        with open(list_configs, mode='a') as f:
            f.write(f'Configuration {i}: {cfg}\n')
        with open(list_results, mode='a') as f:
            f.write(f'Configuration {i}: {cfg}\n')

        # Define the model.
        model = networks.ConvNet(cfg['base_model'], n_classes)

        # Define the transformations and load the data.
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(2),
            model.preprocessor()
        ])
        transform_val = transforms.Compose([model.preprocessor()])
        train_loader = utils.load_data(train_dir,
                                       transform_train,
                                       True,
                                       batch_size=cfg['bsize'])
        val_loader = utils.load_data(val_dir,
                                     transform_val,
                                     False,
                                     batch_size=cfg['bsize'])
        model.to(device)

        # Define the optimizer.
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        criterion.to(device)
        if cfg['optim'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
        elif cfg['optim'] == 'sgd':
            optimizer = optim.SGD(model.parameters(),
                                  lr=cfg['lr'],
                                  momentum=0.9)
        else:
            raise ValueError('The cfg.optim value is not recognized')

        # Train the model.
        train_model(model,
                    train_loader,
                    val_loader,
                    device,
                    criterion,
                    optimizer,
                    trainlog_path,
                    is_binary,
                    n_epochs=50,
                    patience=10)

        # Print confusion matrix and classification report.
        val_preds, val_labels = utils.predictions_and_labels(
            model, val_loader, is_binary, device)
        val_cm = utils.confusion_matrix(val_preds, val_labels)
        val_rep = utils.classification_report(val_preds, val_labels)
        with open(list_results, mode='a') as f:
            f.write('\nValidation data:\n')
            f.write(f'{val_cm}\n')
            f.write(f'{val_rep}\n\n')

        # Save the model.
        if save_weights:
            val_acc = utils.accuracy(val_preds, val_labels)
            model_path = os.path.join(training_path,
                                      f'cfg{i}_val_acc_{val_acc:.3f}.pt')
            torch.save(copy.deepcopy(model.state_dict()), model_path)


def main():
    # Set the number of CPU threads. Uncomment, if your CPU usage is
    # too high.
    # torch.set_num_threads(1)

    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('n_classes', type=int)
    parser.add_argument('train_dir', type=str)
    parser.add_argument('val_dir', type=str)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu_id)  # TODO: Testaa!!!
    train_dir = args.train_dir
    val_dir = args.val_dir

    # Name the grid search directory.
    n_classes = args.n_classes
    time = datetime.datetime.now().strftime('%y-%m-%d-%H%M%S')
    training_path = f'gs_{n_classes}-class_{time}'
    if not os.path.exists(training_path):
        os.mkdir(training_path)

    # Define the hyperparameter grid.
    param_grid = {
        'base_model': [
            'densenet-121', 'mobilenet_v3', 'resnet-50', 'vgg-16',
            'inception_v3'
        ],
        'bsize': [16],
        'optim': ['adam', 'sgd'],
        'lr': [1e-6, 1e-5, 1e-4, 1e-3]
    }
    cfgs = enumerate(list(skl.model_selection.ParameterGrid(param_grid)))

    # Perform the grid search.
    grid_search(cfgs, train_dir, val_dir, training_path, n_classes, device)


if __name__ == '__main__':
    main()

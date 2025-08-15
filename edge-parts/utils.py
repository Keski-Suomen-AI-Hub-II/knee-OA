import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import torch
import torchvision

import image_dataset_with_paths


def load_data_with_paths(data_dir, transform, shuffle, batch_size=16):
    """Return an iterable dataloader including filepaths."""
    dataset = image_dataset_with_paths.ImageDatasetWithPaths(
        data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def load_data(data_dir, transform, shuffle, batch_size=16):
    """Return an iterable dataloader."""
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def predictions_and_labels(
        model, data_loader, is_binary, device=torch.device('cpu')):
    model.eval()
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            if is_binary:
                preds = torch.round(torch.sigmoid(outputs))
            else:
                preds = outputs.argmax(dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return preds_all, labels_all


def accuracy(preds, labels):
    correct = (np.array(preds) == np.array(labels)).sum()
    total = len(preds)
    return correct / total


def confusion_matrix(preds, labels):
    """Return the confusion matrix as a text."""
    return skl.metrics.confusion_matrix(labels, preds)


def confusion_matrix_display(preds, labels, filepath, figsize=(22, 22)):
    """Save a visual confusion matrix."""
    _, ax = plt.subplots(1, 1, figsize=figsize)
    skl.metrics.ConfusionMatrixDisplay.from_predictions(
        preds, labels, xticks_rotation='vertical', ax=ax, colorbar=False)
    plt.savefig(filepath)


def classification_report(preds, labels):
    return skl.metrics.classification_report(labels, preds, digits=4)


def write_learning_curve(cfg_i, history, pdf):
    fig = plt.figure(figsize=(5, 4))
    epochs = history['epoch']
    train_accs = history['train_acc']
    val_accs = history['val_acc']

    plt.plot(epochs, train_accs, label='Learning accuracy')
    plt.plot(epochs, val_accs, label='Validation accuracy')
    plt.title('Learning and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.suptitle(f'Configuration {cfg_i}')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

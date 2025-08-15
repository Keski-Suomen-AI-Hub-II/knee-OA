#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms

import networks
import utils


def test(n_classes, test_dir, weights_path, base_model, res_file):
    model = networks.ConvNet(base_model, n_classes)
    weights = torch.load(weights_path,
                         weights_only=True,
                         map_location=torch.device('cpu'))
    model.load_state_dict(weights)

    transform_test = transforms.Compose([model.preprocessor()])
    test_loader = utils.load_data(test_dir, transform_test, False)

    preds, labels = utils.predictions_and_labels(model, test_loader,
                                                 n_classes == 2)
    test_cm = utils.confusion_matrix(preds, labels)
    test_rep = utils.classification_report(preds, labels)
    with open(res_file, mode='a') as f:
        f.write(f'Classes: {n_classes}\n')
        f.write(f'Base model: {base_model}\n')
        f.write(f'Weights: {weights_path}\n')
        f.write(f'{test_cm}\n')
        f.write(f'{test_rep}\n\n')


def main():
    classnums = [2, 3, 5]
    # FILL IN!
    # Paths to the test data.
    # For example:
    # (2, cropped): 'path/to/test/data/' <-- test data of cropped images,
    #                                        2 classes
    data_dirs = {
        (2, 'cropped'): 'cropped_2-class/test/',
        (3, 'cropped'): 'cropped_3-class/test/',
        (5, 'cropped'): 'cropped_5-class/test/',
        (2, 'non-cropped'): 'non-cropped_2-class/test/',
        (3, 'non-cropped'): 'non-cropped_3-class/test/',
        (5, 'non-cropped'): 'non-cropped_5-class/test/'
    }
    # FILL IN!
    # Paths to the model files.
    weights_paths = {
        (2, 'cropped'): 'path/to/model/file.pt',
        (3, 'cropped'): 'path/to/model/file.pt',
        (5, 'cropped'): 'path/to/model/file.pt',
        (2, 'non-cropped'): 'path/to/model/file.pt',
        (3, 'non-cropped'): 'path/to/model/file.pt',
        (5, 'non-cropped'): 'path/to/model/file.pt'
    }
    # FILL IN!
    # The convnets to use, e.g. 'vgg-16'.
    base_models = {
        (2, 'cropped'): '',
        (3, 'cropped'): '',
        (5, 'cropped'): '',
        (2, 'non-cropped'): '',
        (3, 'non-cropped'): '',
        (5, 'non-cropped'): ''
    }
    res_file = 'test/test_results.txt'

    with open(res_file, mode='w') as f:
        f.write('Test results\n------------\n\n')

    for cropped in ['cropped', 'non-cropped']:
        with open(res_file, mode='a') as f:
            f.write(f'{cropped.upper()}\n')
        for n_classes in classnums:
            test_dir = data_dirs[(n_classes, cropped)]
            weights_path = weights_paths[(n_classes, cropped)]
            base_model = base_models[(n_classes, cropped)]
            test(n_classes, test_dir, weights_path, base_model, res_file)


if __name__ == '__main__':
    main()

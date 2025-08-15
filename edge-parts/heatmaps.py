#!/usr/bin/env python3

import argparse
import os

import skimage as ski
import torch
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget, BinaryClassifierOutputTarget)

import networks
import utils


def load_model(base_model, n_classes, weights_path):
    model = networks.ConvNet(base_model, n_classes)
    if weights_path:
        weights = torch.load(weights_path,
                             weights_only=True,
                             map_location=torch.device('cpu'))
        model.load_state_dict(weights)
    return model


def resize_and_crop(rgb_img, model):
    """Resize and crop the image as in ImageNet preprocessing."""
    resize_l, cropping_l = model.resizing_and_cropping_dims()
    resized = ski.transform.resize(rgb_img, (resize_l, resize_l, 3))
    offset = (resize_l - cropping_l) // 2

    # In case of Inception V3, resize_l - cropping_l is not divisible
    # by 2, so we have to take care of it manually.
    if (resize_l - cropping_l) % 2 == 0:
        cropped = resized[offset:resize_l-offset, offset:resize_l-offset, :]
    else:
        cropped = resized[offset:resize_l -
                          offset-1, offset:resize_l-offset-1, :]
    return cropped


def draw_heatmaps(model, src_rootdir, dst_rootdir, image_weight,
                  is_binary):
    transform = transforms.Compose([model.preprocessor()])
    target_layers = [model.target_layer_for_gradcam()]
    dataloader = utils.load_data_with_paths(
        src_rootdir, transform, False, batch_size=1)
    cam = GradCAM(model, target_layers)

    # Iterate over the data.
    # We need also filepath of each image to load the image.
    # This is because img (below) is preprocessed tensor, i.e.
    # input of the model, and we need also the original image
    # (rgb_img below).
    for fpath, img, label in dataloader:
        fpath = fpath[0]
        label = label.item()
        if is_binary:
            with torch.no_grad():
                predicted_class = torch.round(torch.sigmoid(model(img))).item()
                predicted_class = int(predicted_class)
            targets = [BinaryClassifierOutputTarget(predicted_class)]
        else:
            with torch.no_grad():
                predicted_class = torch.argmax(model(img)).item()
            targets = [ClassifierOutputTarget(predicted_class)]

        # Load the original image. The image is scaled and cropped
        # the same way as in dataloader. This ensures that the heatmap
        # is more accurate because the scaled and cropped image differs
        # slightly from the original image.
        rgb_img = ski.io.imread(fpath)
        rgb_img = ski.color.gray2rgb(rgb_img)
        cropped = resize_and_crop(rgb_img, model)

        # Generate the heatmap.
        grayscale_cam = cam(img, targets=targets)
        visualization = show_cam_on_image(
            cropped,
            grayscale_cam[0, :],
            use_rgb=True,
            image_weight=image_weight)

        # Save the heatmap.
        fname = fpath.split(os.path.sep)[-1]
        dst_fpath = os.path.join(dst_rootdir, str(
            label), f'PRED_{predicted_class}_{fname}')
        ski.io.imsave(dst_fpath, visualization)


def main():
    # Get the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('base_model', type=str)
    parser.add_argument('src_rootdir', type=str)
    parser.add_argument('dst_rootdir', type=str)
    parser.add_argument('--weights_path', type=str, default='')
    parser.add_argument('--image_weight', type=float, default=.8)
    args = parser.parse_args()

    # Get number of the classes and make a destination directory
    # for each class.
    n_classes = len(os.listdir(args.src_rootdir))
    is_binary = n_classes == 2
    print(f'Number of classes found: {n_classes}')
    for class_number in range(n_classes):
        os.makedirs(os.path.join(args.dst_rootdir,
                    str(class_number)), exist_ok=True)

    # Draw and save heatmaps.
    model = load_model(args.base_model, n_classes, args.weights_path)
    draw_heatmaps(model, args.src_rootdir, args.dst_rootdir,
                  args.image_weight, is_binary)


if __name__ == '__main__':
    main()

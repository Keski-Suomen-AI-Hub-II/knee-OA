#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from matplotlib import colormaps
from skimage.color import gray2rgb
from skimage.io import imread


def generate_heatmap(model, img1_path, img2_path, img_size, num_of_img, alpha):
    """Return the prediction and heatmap."""
    # Based on the code by Francois Chollet in Deep Learning with Python
    # (2nd ed.), see also https://keras.io/examples/vision/grad_cam/.

    # Load images and reshape them as batches of one image.
    im1 = gray2rgb(imread(img1_path).astype('float32'))
    im2 = gray2rgb(imread(img2_path).astype('float32'))
    im1_as_batch = im1.reshape((1, img_size[0], img_size[1], 3))
    im2_as_batch = im2.reshape((1, img_size[0], img_size[1], 3))

    # The convolutional branches as models.
    last_conv_layer_model1 = tf.keras.Model(
        model.get_layer('branch1').inputs,
        model.get_layer('branch1').output)
    last_conv_layer_model2 = tf.keras.Model(
        model.get_layer('branch2').inputs,
        model.get_layer('branch2').output)

    # The classifier head as a model.
    conv_output_shape = model.get_layer('branch1').output_shape[1:]
    classifier_input1 = tf.keras.Input(shape=conv_output_shape)
    classifier_input2 = tf.keras.Input(shape=conv_output_shape)
    # It is assumed that the classifier consists of 4 layers.
    x = model.get_layer(index=-4)([classifier_input1, classifier_input2])
    for ind in range(-3, 0):
        x = model.get_layer(index=ind)(x)
    classifier_model = tf.keras.Model([classifier_input1, classifier_input2],
                                      x)
    # Remove the activation of the Dense layer, because in the
    # original article, the gradients are computed before the
    # softmax.
    classifier_model.get_layer(index=-1).activation = None

    # Get gradients.
    with tf.GradientTape() as tape:
        last_conv_layer_output1 = last_conv_layer_model1(im1_as_batch,
                                                         training=False)
        last_conv_layer_output2 = last_conv_layer_model2(im2_as_batch,
                                                         training=False)
        last_conv_layer_output = [
            last_conv_layer_output1, last_conv_layer_output2
        ][num_of_img - 1]
        tape.watch(last_conv_layer_output)
        preds = classifier_model(
            [last_conv_layer_output1, last_conv_layer_output2], training=False)
        prediction = np.argmax(preds)
        top_class_channel = preds[:, prediction]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # Get the heatmap.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    jet = colormaps.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    im = [im1, im2][num_of_img - 1]
    jet_heatmap = jet_heatmap.resize((im.shape[1], im.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed = alpha * jet_heatmap + im
    superimposed = tf.keras.utils.array_to_img(superimposed)

    return prediction, superimposed


def save_heatmaps(model, img1_path, img2_path, img_size, heatmaps_path,
                  classname, alpha):
    """Generate and save heatmaps into heatmaps_path."""
    fig = plt.figure(figsize=(10, 4))
    for i in [1, 2]:
        prediction, superimposed = generate_heatmap(model, img1_path,
                                                    img2_path, img_size, i,
                                                    alpha)
        plt.subplot(1, 2, i)
        plt.imshow(superimposed)
    plt.suptitle('Class: {}\nPrediction: {}'.format(classname, prediction))
    plt.savefig(heatmaps_path)
    plt.close(fig)


def heatmaps_from_dirs(model, src_dir1, src_dir2, img_size, dst_dir, alpha=.7):
    """Generate heatmaps from the images inside src_dir1 and src_dir2."""
    for classname in os.listdir(src_dir1):
        src_classpath1 = os.path.sep.join([src_dir1, classname])
        src_classpath2 = os.path.sep.join([src_dir2, classname])
        dst_classpath = os.path.sep.join([dst_dir, classname])
        os.makedirs(dst_classpath, exist_ok=True)
        filenames = os.listdir(src_classpath1)
        for filename in filenames:
            src_filepath1 = os.path.sep.join([src_classpath1, filename])
            src_filepath2 = os.path.sep.join([src_classpath2, filename])
            dst_filepath = os.path.sep.join([dst_classpath, filename])
            save_heatmaps(model, src_filepath1, src_filepath2, img_size,
                          dst_filepath, classname, alpha)


def main():
    pass


if __name__ == '__main__':
    main()

import os

import numpy as np
import tensorflow as tf
from skimage.exposure import rescale_intensity
from skimage.feature import match_template
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from tensorflow.config import list_physical_devices, set_visible_devices
from tensorflow.config.experimental import set_memory_growth


def reserve_gpu(id):
    if id < 0:
        return
    gpus = list_physical_devices('GPU')
    if gpus:
        gpu = gpus[id]
        set_visible_devices(gpu, 'GPU')
        set_memory_growth(gpu, True)


def write_confusion_matrix(model, data, filepath, desc_text, multiclass=True):
    """Save confusion matrix to a given filepath."""
    labels = np.concatenate([label for _, label in data], axis=0)
    labels = tf.math.argmax(labels, axis=-1)
    preds = model.predict(data)
    preds = tf.math.argmax(preds, axis=-1)
    cm = confusion_matrix(labels, preds)
    with open(filepath, mode='a') as f:
        f.write(desc_text)
        f.write(str(cm))
        f.write('\n')
        for i in range(cm.shape[0]):
            class_acc = cm[i, i] / sum(cm[i])
            f.write('Class {}: {:.6f}\n'.format(i, class_acc))
        f.write('\n')


def enhance_images(src_path, dst_path):
    """Perform contrast stretching on all the images inside src_path."""
    for src_dirpath, _, filenames in os.walk(src_path):
        dst_dirpath = src_dirpath.replace(src_path, dst_path, 1)
        os.makedirs(dst_dirpath, exist_ok=True)
        for filename in filenames:
            src_filepath = os.path.sep.join([src_dirpath, filename])
            dst_filepath = os.path.sep.join([dst_dirpath, filename])
            img = imread(src_filepath)
            img_enh = rescale_intensity(img)
            imsave(dst_filepath, img_enh)


def read_templates(templates_path):
    """Return a list of templates."""
    templates = []
    for name in os.listdir(templates_path):
        templ = imread(os.path.sep.join([templates_path, name]))
        templates.append(templ)
    return templates


def extract_eminentia(path_to_img, eminentia_templates, crop_before=False):
    """"Return the best matching eminentia part."""
    matches = []
    img = imread(path_to_img)

    # Only consider the center of the image.
    if crop_before:
        y_part = int(img.shape[0] / 4)
        x_part = int(img.shape[1] / 4)
        img = img[y_part:3 * y_part, x_part:3 * x_part]

    # Find the best matching template.
    for template in eminentia_templates:
        match = match_template(img, template)
        matches.append(match.max())
    index = np.argmax(matches)
    em = eminentia_templates[index]
    res = match_template(img, em)

    # Extract and return.
    x, y = np.unravel_index(np.argmax(res), res.shape)
    em_width, em_height = em.shape
    extracted = img[y:y + em_height, x:x + em_width]
    return extracted


def crop_images(datapath, cropped_datapath, templates_path, shape_to_save):
    """Crop images, then resize and save the cropped."""
    templates = read_templates(templates_path)
    for dirpath, _, filenames in os.walk(datapath):
        cropped_dirpath = dirpath.replace(datapath, cropped_datapath, 1)
        os.makedirs(cropped_dirpath, exist_ok=True)
        for filename in filenames:
            filepath = os.path.sep.join([dirpath, filename])
            cropped_filepath = os.path.sep.join([cropped_dirpath, filename])
            cropped = extract_eminentia(filepath, templates, crop_before=True)
            cropped_resized = resize(cropped,
                                     shape_to_save,
                                     preserve_range=True).astype('uint8')
            imsave(cropped_filepath, cropped_resized)


def main():
    templates_path = 'data_eminentias'
    datapath = 'data'
    cropped_datapath = 'data_autocropped'
    shape_to_save = (224, 224)
    crop_images(datapath, cropped_datapath, templates_path, shape_to_save)
    enhance_images(cropped_datapath, 'hcontr_data_autocropped')


if __name__ == '__main__':
    main()

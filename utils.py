import os

import numpy as np
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.feature import match_template
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.utils import shuffle as sk_shuffle
from tensorflow.config import list_physical_devices, set_visible_devices
from tensorflow.config.experimental import set_memory_growth
from tensorflow.data import Dataset


def load_data(dirname1, dirname2, shape, random_seed=None):
    """Load parallel data from the two directories."""
    # Initialize two arrays for images.
    # Despite the original shape, one dimension is reserved for 3
    # channels.
    n_imgs = sum([len(names) for _, _, names in os.walk(dirname1)])
    arr1 = np.zeros(shape=(n_imgs, shape[0], shape[1], 3), dtype='float32')
    arr2 = np.zeros(shape=(n_imgs, shape[0], shape[1], 3), dtype='float32')

    # Initialize array for labels. One-hot encoding is used.
    classnames = os.listdir(dirname1)
    classes = len(classnames)
    labels = np.zeros(shape=(n_imgs, classes), dtype='float32')

    # Fill the arrays.
    # Grayscale images are converted to RGB.
    convert_to_rgb = len(shape) < 3
    i = 0
    for classname in classnames:
        classpath1 = os.path.sep.join([dirname1, classname])
        classpath2 = os.path.sep.join([dirname2, classname])
        filenames = os.listdir(classpath1)
        for filename in filenames:
            filepath1 = os.path.sep.join([classpath1, filename])
            filepath2 = os.path.sep.join([classpath2, filename])
            img_arr1 = imread(filepath1)
            img_arr2 = imread(filepath2)
            if convert_to_rgb:
                img_arr1 = gray2rgb(img_arr1)
                img_arr2 = gray2rgb(img_arr2)
            arr1[i] = img_arr1
            arr2[i] = img_arr2
            classnum = int(classname)
            labels[i, classnum] = 1.0
            i += 1
    # Shuffle. Random seed is used to ensure that all the arrays
    # are shuffled consistently.
    if random_seed is not None:
        arr1 = sk_shuffle(arr1, random_state=random_seed)
        arr2 = sk_shuffle(arr2, random_state=random_seed)
        labels = sk_shuffle(labels, random_state=random_seed)

    return arr1, arr2, labels


def load_as_dataset(dirname1,
                    dirname2,
                    shape,
                    batch_size,
                    random_seed=None,
                    input1_name='input1',
                    input2_name='input2'):
    """Load parallel data from the directories and convert to dataset."""
    arr1, arr2, labels = load_data(dirname1, dirname2, shape, random_seed)
    ds = (Dataset.from_tensor_slices(({
        input1_name: arr1,
        input2_name: arr2
    }, labels))).batch(batch_size)
    return ds


def reserve_gpu(id):
    gpus = list_physical_devices('GPU')
    if gpus:
        gpu = gpus[id]
        set_visible_devices(gpu, 'GPU')
        set_memory_growth(gpu, True)


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

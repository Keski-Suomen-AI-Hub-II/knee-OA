#!/usr/bin/env python3

import argparse
import os

import numpy as np
import skimage as ski


def enhance_contrast(img):
    enh = ski.exposure.equalize_hist(img)
    return ski.util.img_as_ubyte(enh)


def crop_and_merge(img):
    img1 = img[24:, 0:50]
    img2 = img[24:, 174:]
    merged = np.concatenate((img1, img2), axis=1)
    return merged


def process_the_dirs(src_rootdir, dst_rootdir1, dst_rootdir2):
    """Save the enhanced to dst_rootdir1 and cropped to rootdir2."""
    for src_dirpath, _, fnames in os.walk(src_rootdir):
        dst_dirpath1 = os.path.abspath(src_dirpath).replace(os.path.abspath(
            src_rootdir), os.path.abspath(dst_rootdir1), 1)
        dst_dirpath2 = os.path.abspath(src_dirpath).replace(os.path.abspath(
            src_rootdir), os.path.abspath(dst_rootdir2), 1)
        os.makedirs(dst_dirpath1, exist_ok=True)
        os.makedirs(dst_dirpath2, exist_ok=True)
        for fname in fnames:
            src_fpath = os.path.join(src_dirpath, fname)
            dst_fpath1 = os.path.join(dst_dirpath1, fname)
            dst_fpath2 = os.path.join(dst_dirpath2, fname)
            # Save the enhanced image.
            img = ski.io.imread(src_fpath)
            enhanced = enhance_contrast(img)
            ski.io.imsave(dst_fpath1, enhanced)
            # Save the cropped image.
            cropped = crop_and_merge(enhanced)
            ski.io.imsave(dst_fpath2, cropped)


def main():
    # Get the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('src_rootdir', type=str)
    parser.add_argument('dst_rootdir1', type=str)
    parser.add_argument('dst_rootdir2', type=str)
    args = parser.parse_args()

    process_the_dirs(args.src_rootdir, args.dst_rootdir1, args.dst_rootdir2)


if __name__ == '__main__':
    main()

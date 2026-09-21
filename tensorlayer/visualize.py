# -*- coding: utf-8 -*-
"""
Visualization and image I/O (PyTorch equivalent of tensorlayer.visualize)
"""

import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image as _torch_save_image


def read_image(image, path=''):
    """Read one image.

    Parameters
    ----------
    image : str
        The image file name.
    path : str
        The image folder path.

    Returns
    -------
    numpy.array
        The image with shape [H, W, C].
    """
    img = Image.open(os.path.join(path, image)).convert('RGB')
    return np.array(img)


def read_images(img_list, path='', n_threads=10, printable=True):
    """Returns all images in list by given path and name of each image file.

    Parameters
    ----------
    img_list : list of str
        The image file names.
    path : str
        The image folder path.
    n_threads : int
        The number of threads to read image.
    printable : bool

    Returns
    -------
    list of numpy.array
        The images.
    """
    from .prepro import threading_data
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx:idx + n_threads]
        b_imgs = threading_data(b_imgs_list, fn=read_image, path=path)
        imgs.extend(b_imgs)
        if printable:
            print('[*] read %d from %s' % (len(imgs), path))
    return imgs


def save_image(image, image_path=''):
    """Save a single image.

    Parameters
    ----------
    image : numpy array
        [H, W, C] or [H, W].
    image_path : str
    """
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    img.save(image_path)


def save_images(images, size, image_path=''):
    """Save multiple images in a grid.

    Parameters
    ----------
    images : numpy array
        [N, H, W, C] or [N, H, W].
    size : list of int
        [row, col] for the grid.
    image_path : str
    """
    if images.max() <= 1.0:
        images = (images * 255).clip(0, 255).astype(np.uint8)

    h, w = images.shape[1], images.shape[2]
    c = images.shape[3] if len(images.shape) == 4 else 1
    row, col = size

    if c == 1:
        canvas = np.zeros((row * h, col * w), dtype=np.uint8)
        for idx, img in enumerate(images[:row * col]):
            r = idx // col
            c_idx = idx % col
            canvas[r * h:(r + 1) * h, c_idx * w:(c_idx + 1) * w] = img[:, :, 0] if img.ndim == 3 else img
    else:
        canvas = np.zeros((row * h, col * w, c), dtype=np.uint8)
        for idx, img in enumerate(images[:row * col]):
            r = idx // col
            c_idx = idx % col
            canvas[r * h:(r + 1) * h, c_idx * w:(c_idx + 1) * w] = img

    img = Image.fromarray(canvas)
    img.save(image_path)

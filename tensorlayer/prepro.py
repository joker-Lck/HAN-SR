# -*- coding: utf-8 -*-
"""
Data preprocessing utilities (PyTorch equivalent of tensorlayer.prepro)
"""

import threading
import numpy as np
from PIL import Image


def threading_data(data, fn, thread_count=None, **kwargs):
    """Process a batch of data by given function by threading.

    Parameters
    ----------
    data : list or numpy.array
        The data to be processed.
    fn : function
        The function for data processing.
    thread_count : int
        The number of threads to use. Default is len(data).

    Returns
    -------
    list
        Processed data.

    Examples
    --------
    >>> images = tl.prepro.threading_data(images, tl.prepro.crop, wrg=384, hrg=384, is_random=True)
    """
    if thread_count is None:
        thread_count = len(data)

    results = [None] * len(data)

    def worker(idx, x):
        results[idx] = fn(x, **kwargs)

    threads = []
    for idx, x in enumerate(data):
        t = threading.Thread(target=worker, args=(idx, x))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if isinstance(data, np.ndarray):
        return np.array(results)
    return results


def crop(x, wrg=384, hrg=384, is_random=True):
    """Crop an image.

    Parameters
    ----------
    x : numpy array
        Image with shape [H, W, C].
    wrg : int
        Width of the crop.
    hrg : int
        Height of the crop.
    is_random : bool
        Whether to crop randomly.

    Returns
    -------
    numpy array
        Cropped image.
    """
    h, w = x.shape[0], x.shape[1]
    if is_random:
        top = np.random.randint(0, max(h - hrg, 1))
        left = np.random.randint(0, max(w - wrg, 1))
    else:
        top = max((h - hrg) // 2, 0)
        left = max((w - wrg) // 2, 0)
    return x[top:top + hrg, left:left + wrg].copy()


def imresize(x, size=[96, 96], interp='bicubic', mode=None):
    """Resize an image.

    Parameters
    ----------
    x : numpy array
        Image with shape [H, W, C] or [H, W].
    size : list of int
        Target size [H, W].
    interp : str
        Interpolation method.

    Returns
    -------
    numpy array
        Resized image.
    """
    img = Image.fromarray(x.astype(np.uint8))
    method = Image.BICUBIC if interp == 'bicubic' else Image.BILINEAR
    img = img.resize((size[1], size[0]), method)
    return np.array(img).astype(x.dtype)


def flip_axis(x, axis=0, is_random=True):
    """Flip an image along an axis.

    Parameters
    ----------
    x : numpy array
    axis : int
        0 for vertical, 1 for horizontal.
    is_random : bool
    """
    if is_random:
        if np.random.random() < 0.5:
            x = np.flip(x, axis=axis).copy()
    else:
        x = np.flip(x, axis=axis).copy()
    return x


def rotation(x, rg=20, is_random=True, row_index=0, col_index=1, channel_index=2):
    """Rotate an image.

    Parameters
    ----------
    x : numpy array
    rg : int
        Rotation range in degrees.
    is_random : bool
    """
    from scipy.ndimage import rotate
    if is_random:
        theta = np.random.uniform(-rg, rg)
    else:
        theta = rg
    return rotate(x, theta, reshape=False).astype(x.dtype)

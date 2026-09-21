# -*- coding: utf-8 -*-
"""
File I/O utilities (PyTorch equivalent of tensorlayer.files)
Save and restore model, save and load variables.
"""

import os
import re
import numpy as np
import torch


def exists_or_mkdir(path):
    """Create folder if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print('[*] created folder: %s' % path)
    return os.path.exists(path)


def load_file_list(path, regx='.*.png', printable=True):
    """Load file list from a folder by given regular expression.

    Parameters
    ----------
    path : str
        The folder path.
    regx : str
        The regular expression for matching files.
    printable : bool
        Whether to print the number of files found.

    Returns
    -------
    list of str
        The file paths.

    """
    file_list = []
    for f in os.listdir(path):
        if re.search(regx, f):
            file_list.append(os.path.join(path, f))
    file_list.sort()
    if printable:
        print('[*] load_file_list: %d files found in %s' % (len(file_list), path))
    return file_list


def save_npz(save_list=None, name='model.npz', sess=None):
    """Save parameters as .npz file (PyTorch: save state_dict).

    Parameters
    ----------
    save_list : list of parameters or state_dict
    name : str
        File name.
    """
    if isinstance(save_list, dict):
        torch.save(save_list, name)
    else:
        torch.save(save_list, name)
    print('[*] saved %s' % name)


def load_npz(path='model.npz'):
    """Load parameters from .npz file.

    Returns
    -------
    state_dict or list
    """
    return torch.load(path, map_location='cpu')


def assign_params(sess, params, network):
    """Assign parameters to a network (PyTorch: load_state_dict).

    Parameters
    ----------
    params : dict or list
        Parameters to assign.
    network : nn.Module
        The target network.
    """
    if isinstance(params, dict):
        network.load_state_dict(params, strict=False)
    elif isinstance(params, list):
        # assign_params from a list of tensors (VGG style)
        param_dict = {}
        for i, (name, _) in enumerate(network.named_parameters()):
            if i < len(params):
                param_dict[name] = torch.from_numpy(params[i]) if isinstance(params[i], np.ndarray) else params[i]
        network.load_state_dict(param_dict, strict=False)


def load_and_assign_npz(sess=None, name='model.npz', network=None):
    """Load .npz file and assign to network.

    Returns
    -------
    bool
        True if successful, False if file not found.
    """
    if not os.path.exists(name):
        print('[!] load_and_assign_npz: %s not found' % name)
        return False
    params = torch.load(name, map_location='cpu')
    if isinstance(params, dict):
        network.load_state_dict(params, strict=False)
    print('[*] loaded %s' % name)
    return True


def get_variables_with_name(name, train_only=True, printable=True):
    """Get variables by name (PyTorch: filter named_parameters).

    Parameters
    ----------
    name : str
        The name pattern to match.
    train_only : bool
        Whether to only return trainable parameters.
    printable : bool
        Whether to print the number of variables found.

    Returns
    -------
    list of nn.Parameter
    """
    # This function is called with a model reference, not directly
    # In PyTorch, we handle this differently - see train.py
    return []

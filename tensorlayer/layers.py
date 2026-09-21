# -*- coding: utf-8 -*-
"""
Layer utilities (PyTorch equivalent of tensorlayer.layers)
"""

import torch


def get_variables_with_name(model, name='', train_only=True, printable=True):
    """Get variables by name pattern.

    Parameters
    ----------
    model : nn.Module
        The model to search.
    name : str
        The name pattern to match.
    train_only : bool
        Whether to only return trainable parameters.
    printable : bool

    Returns
    -------
    list of nn.Parameter
    """
    params = []
    for param_name, param in model.named_parameters():
        if name in param_name:
            if train_only and not param.requires_grad:
                continue
            params.append(param)
    if printable:
        print('[*] get_variables_with_name: %d params found with name "%s"' % (len(params), name))
    return params


def initialize_global_variables(sess=None):
    """Initialize global variables (PyTorch: no-op, handled by model.to(device))."""
    pass


def print_params(model, details=True):
    """Print model parameters.

    Parameters
    ----------
    model : nn.Module
    details : bool
    """
    total = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        total += numel
        if details:
            print('  %s: %s' % (name, str(param.shape)))
    print('Total params: {:,}'.format(total))

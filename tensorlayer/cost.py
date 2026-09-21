# -*- coding: utf-8 -*-
"""
Loss functions (PyTorch equivalent of tensorlayer.cost)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_cross_entropy(logits, targets, name=None):
    """Sigmoid cross-entropy loss.

    Parameters
    ----------
    logits : Tensor
        Predicted logits.
    targets : Tensor
        Target labels (0 or 1).

    Returns
    -------
    Tensor
        Loss value.
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def mean_squared_error(output, target, is_mean=True):
    """Mean squared error loss.

    Parameters
    ----------
    output : Tensor
    target : Tensor
    is_mean : bool
        Whether to return mean.

    Returns
    -------
    Tensor
    """
    if is_mean:
        return F.mse_loss(output, target)
    else:
        return F.mse_loss(output, target, reduction='sum')


def cross_entropy(output, target):
    """Softmax cross-entropy loss.

    Parameters
    ----------
    output : Tensor
        Logits.
    target : Tensor
        Class indices.

    Returns
    -------
    Tensor
    """
    return F.cross_entropy(output, target)


def l1_loss(output, target):
    """L1 (MAE) loss.

    Parameters
    ----------
    output : Tensor
    target : Tensor

    Returns
    -------
    Tensor
    """
    return F.l1_loss(output, target)

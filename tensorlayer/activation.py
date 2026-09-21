# -*- coding: utf-8 -*-
"""
Activation functions (PyTorch equivalent of tensorlayer.activation)
"""

import torch.nn.functional as F


def lrelu(x, alpha=0.2):
    """Leaky ReLU.

    Parameters
    ----------
    x : Tensor
    alpha : float
        Slope for negative values.

    Returns
    -------
    Tensor
    """
    return F.leaky_relu(x, negative_slope=alpha)


def identity(x):
    """Identity activation."""
    return x

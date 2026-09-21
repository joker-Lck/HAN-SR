# -*- coding: utf-8 -*-
"""
General utilities (PyTorch equivalent of tensorlayer.utils)
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=0):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

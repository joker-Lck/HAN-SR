"""PyTorch equivalent of tensorlayer for SEFNet"""
from __future__ import absolute_import

import os
import glob as _glob

from . import activation
from . import cost
from . import files
from . import layers
from . import prepro
from . import utils
from . import visualize

# alias
act = activation
vis = visualize

__version__ = "1.0.0"

global_flag = {}
global_dict = {}

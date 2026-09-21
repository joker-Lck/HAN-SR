#! /usr/bin/python
# -*- coding: utf8 -*-
"""
SEFNet Model Definitions
网络定义集中管理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture import RRDB_Net, Discriminator_VGG, VGGFeatureExtractor
from block import (MultiScaleExposureConsistencyLoss, compute_luminance,
                   relativistic_discriminator_loss, relativistic_generator_loss)
from data import SRDataset

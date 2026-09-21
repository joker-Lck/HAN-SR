"""
SEFNet - Data Loading & Augmentation
数据集加载与数据增强
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SRDataset(Dataset):
    """
    超分辨率数据集

    支持两种数据组织方式:
    1. 直接文件夹模式: hr_path下直接放HR图片，自动双三次下采样生成LR
    2. 配对文件夹模式: hr_path和lr_path分别放HR和LR图片
    """
    def __init__(self, hr_path, lr_path=None, crop_size=128, scale_factor=4,
                 augment=True, repeat=1):
        super(SRDataset, self).__init__()
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.lr_crop_size = crop_size // scale_factor

        self.hr_images = sorted([
            os.path.join(hr_path, f) for f in os.listdir(hr_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ])

        if lr_path is not None:
            self.lr_images = sorted([
                os.path.join(lr_path, f) for f in os.listdir(lr_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
            ])
            assert len(self.hr_images) == len(self.lr_images), \
                "HR and LR image counts must match"
        else:
            self.lr_images = None

        self.total_len = len(self.hr_images) * repeat

    def __len__(self):
        return self.total_len

    def _random_crop(self, hr_img, lr_img):
        w, h = hr_img.size
        th, tw = self.crop_size, self.crop_size

        if w < tw or h < th:
            hr_img = hr_img.resize((max(w, tw), max(h, th)), Image.BICUBIC)
            w, h = hr_img.size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        hr_patch = hr_img.crop((j, i, j + tw, i + th))

        if self.lr_images is None:
            lr_patch = hr_patch.resize(
                (self.lr_crop_size, self.lr_crop_size), Image.BICUBIC
            )
        else:
            lr_i = i // self.scale_factor
            lr_j = j // self.scale_factor
            lr_patch = lr_img.crop((
                lr_j, lr_i,
                lr_j + self.lr_crop_size,
                lr_i + self.lr_crop_size
            ))

        return lr_patch, hr_patch

    def _augment(self, lr_img, hr_img):
        """数据增强: 随机翻转和旋转"""
        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.ROTATE_90)
            hr_img = hr_img.transpose(Image.ROTATE_90)
        return lr_img, hr_img

    def __getitem__(self, idx):
        idx = idx % len(self.hr_images)

        hr_img = Image.open(self.hr_images[idx]).convert('RGB')

        if self.lr_images is not None:
            lr_img = Image.open(self.lr_images[idx]).convert('RGB')
        else:
            lr_img = None

        # 裁剪
        lr_patch, hr_patch = self._random_crop(hr_img, lr_img)

        # 数据增强
        if self.augment:
            lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

        # 转tensor，归一化到[0, 1]
        to_tensor = transforms.ToTensor()
        lr_tensor = to_tensor(lr_patch)
        hr_tensor = to_tensor(hr_patch)

        return lr_tensor, hr_tensor


class ExposureGradientSynthesizer:
    """
    曝光渐变数据合成器

    从正常曝光图片合成曝光渐变图片，用于训练SEFNet
    随机选择渐变方向和强度，模拟真实世界的曝光不均匀场景
    """
    def __init__(self, exposure_range=(0.3, 1.5)):
        self.exposure_range = exposure_range
        self.directions = ['top_bottom', 'left_right', 'center_edge']

    def __call__(self, img_tensor):
        """
        img_tensor: [C, H, W] 归一化到[0,1]
        返回: (degraded, grad) degraded为渐变曝光后的图片，grad为曝光增益图
        """
        C, H, W = img_tensor.shape
        direction = random.choice(self.directions)

        min_exp = random.uniform(self.exposure_range[0], 0.8)
        max_exp = random.uniform(1.0, self.exposure_range[1])

        if direction == 'top_bottom':
            grad = torch.linspace(max_exp, min_exp, H).unsqueeze(0).unsqueeze(2)
            grad = grad.expand(C, H, W)
        elif direction == 'left_right':
            grad = torch.linspace(max_exp, min_exp, W).unsqueeze(0).unsqueeze(1)
            grad = grad.expand(C, H, W)
        elif direction == 'center_edge':
            y = torch.linspace(-1, 1, H).unsqueeze(1).expand(H, W)
            x = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
            dist = torch.sqrt(x ** 2 + y ** 2) / 1.414
            grad = (max_exp - min_exp) * (1 - dist) + min_exp
            grad = grad.unsqueeze(0).expand(C, H, W)
        else:
            grad = torch.ones(C, H, W)

        degraded = img_tensor * grad
        degraded = torch.clamp(degraded, 0, 1)

        return degraded, grad[0:1]  # 返回亮度通道作为曝光GT

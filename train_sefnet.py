#! /usr/bin/python
# -*- coding: utf8 -*-
"""
SEFNet Training Script - 可直接运行的完整训练脚本

功能:
  1. 自动从任意图片目录生成曝光渐变训练数据
  2. 三阶段训练: 内容预热 → 曝光一致性联合训练 → GAN对抗训练
  3. 自动保存样张、checkpoint、训练日志

使用方式:
  # 用测试图作为演示（小数据集，快速验证流程）
  python train_sefnet.py --hr_dir ./test_images --stage1_epochs 5 --stage2_epochs 5 --stage3_epochs 5

  # 用自己的图片目录（推荐放100+张图）
  python train_sefnet.py --hr_dir /path/to/images --stage1_epochs 50 --stage2_epochs 200 --stage3_epochs 100

  # 用DIV2K数据集（正式训练）
  python train_sefnet.py --hr_dir data/DIV2K_train_HR --stage1_epochs 50 --stage2_epochs 200 --stage3_epochs 100

  # 仅跑某一阶段
  python train_sefnet.py --hr_dir data/DIV2K_train_HR --stage 1
"""

import os
import sys
import time
import glob
import random
import argparse
import logging
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# ===== 项目模块导入 =====
import architecture as arch
import block as B
from config import config
from utils import save_checkpoint, load_checkpoint, ARCH_DEFAULTS

# ===== 日志 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('SEFNet')


# ====================================================================
# 第一部分: 曝光退化数据合成（训练时在线生成，不需要预处理）
# 支持三种曝光模式: 渐变曝光、普通曝光、混合曝光
# ====================================================================

class ExposureSynthesizer:
    """
    在线曝光退化合成器（增强版）

    三类曝光退化:
    1. 渐变曝光 (gradient): 亮度沿空间方向渐变（原有6种类型）
    2. 普通曝光 (uniform): 全图均匀过曝/欠曝（新增）
    3. 混合曝光 (mixed): 全局偏移 + 局部渐变叠加（新增）

    采样比例: 渐变50% + 普通30% + 混合20%
    原因: 普通曝光是真实场景中最常见的退化类型，不能只训渐变。
    """

    def __init__(self, gain_range=(0.2, 3.0), blur_range=(51, 201)):
        self.gain_range = gain_range
        self.blur_range = blur_range
        # 三种曝光模式及采样权重
        self.modes = ['gradient', 'uniform', 'mixed']
        self.mode_weights = [0.50, 0.30, 0.20]
        # 渐变子类型
        self.grad_types = ['vertical', 'horizontal', 'radial_center',
                           'radial_edge', 'diagonal', 'irregular']
        self.grad_weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]

    def __call__(self, img):
        """
        img: numpy array [H, W, 3], uint8, RGB
        返回: (degraded, gain_map, exposure_type)
            degraded: [H, W, 3] uint8
            gain_map: [H, W] float32
            exposure_type: str, 'gradient'/'uniform'/'mixed'
        """
        h, w = img.shape[:2]
        mode = random.choices(self.modes, weights=self.mode_weights, k=1)[0]

        if mode == 'uniform':
            gain_map = self._make_uniform_gain(h, w)
            exposure_type = 'uniform'
        elif mode == 'mixed':
            gain_map = self._make_mixed_gain(h, w)
            exposure_type = 'mixed'
        else:
            gain_map = self._make_gradient_gain(h, w)
            exposure_type = 'gradient'

        img_f = img.astype(np.float32)
        degraded = np.clip(img_f * gain_map[:, :, np.newaxis], 0, 255).astype(np.uint8)
        return degraded, gain_map.astype(np.float32), exposure_type

    def _make_uniform_gain(self, h, w):
        """
        普通曝光: 全图均匀增益，模拟相机曝光参数不当
        欠曝 gain < 1.0，过曝 gain > 1.0
        """
        gain = random.uniform(self.gain_range[0], self.gain_range[1])
        return np.full((h, w), gain, dtype=np.float32)

    def _make_gradient_gain(self, h, w):
        """渐变曝光: 空间方向渐变（原有逻辑）"""
        grad_type = random.choices(self.grad_types, weights=self.grad_weights, k=1)[0]
        intensity = random.uniform(self.gain_range[0], self.gain_range[1])
        reverse = random.random() > 0.5
        gain_map = self._compute_gradient(h, w, grad_type, intensity, reverse)
        # 平滑
        k = random.choice(range(self.blur_range[0], self.blur_range[1] + 1, 2))
        gain_map = cv2.GaussianBlur(gain_map, (k, k), 0)
        return gain_map

    def _make_mixed_gain(self, h, w):
        """
        混合曝光: 全局偏移 + 局部渐变叠加
        模拟真实场景: 如室内拍照时窗户过曝+角落欠曝
        """
        # 全局偏移（较小范围）
        global_gain = random.uniform(0.6, 1.6)
        # 局部渐变（较小强度）
        grad_type = random.choices(self.grad_types[:4], k=1)[0]  # 只用基础方向
        local_intensity = random.uniform(0.5, 1.5)
        reverse = random.random() > 0.5
        local_gain = self._compute_gradient(h, w, grad_type, local_intensity, reverse)
        # 平滑
        k = random.choice(range(self.blur_range[0], self.blur_range[1] + 1, 2))
        local_gain = cv2.GaussianBlur(local_gain, (k, k), 0)
        # 叠加: 全局偏移 × 局部渐变
        return (global_gain * local_gain).astype(np.float32)

    def _compute_gradient(self, h, w, grad_type, intensity, reverse):
        if grad_type == 'vertical':
            g = np.linspace(1.0, intensity, h) if not reverse else np.linspace(intensity, 1.0, h)
            return np.tile(g.reshape(-1, 1), (1, w))
        elif grad_type == 'horizontal':
            g = np.linspace(1.0, intensity, w) if not reverse else np.linspace(intensity, 1.0, w)
            return np.tile(g.reshape(1, -1), (h, 1))
        elif grad_type == 'radial_center':
            cy, cx = h / 2, w / 2
            Y, X = np.mgrid[0:h, 0:w]
            dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
            if reverse:
                return 1.0 + (intensity - 1.0) * (1 - np.cos(dist * np.pi / 2))
            else:
                return intensity - (intensity - 1.0) * (1 - np.cos(dist * np.pi / 2))
        elif grad_type == 'radial_edge':
            cy, cx = h / 2, w / 2
            Y, X = np.mgrid[0:h, 0:w]
            dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
            if reverse:
                return 1.0 + (intensity - 1.0) * np.cos(dist * np.pi / 2)
            else:
                return intensity - (intensity - 1.0) * np.cos(dist * np.pi / 2)
        elif grad_type == 'diagonal':
            Y, X = np.mgrid[0:h, 0:w]
            diag = (Y / h + X / w) / 2
            if reverse:
                return 1.0 + (intensity - 1.0) * diag
            else:
                return intensity - (intensity - 1.0) * diag
        elif grad_type == 'irregular':
            sh, sw = max(4, h // 64), max(4, w // 64)
            noise = np.random.randn(sh, sw).astype(np.float32)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            return 1.0 + (intensity - 1.0) * noise
        return np.ones((h, w), dtype=np.float32)


# ====================================================================
# 第二部分: 训练数据集（在线生成曝光渐变 + 真实退化）
# ====================================================================

class SEFNetTrainDataset(Dataset):
    """
    SEFNet 训练数据集（增强版）

    数据流:
    HR图 → 随机裁剪 → 随机翻转旋转 → 曝光退化合成 → 真实退化下采样 → LR输入
                                     → 正常曝光HR → 监督目标(GT)

    支持:
    - 任意图片目录作为HR源
    - 在线生成曝光退化（渐变+普通+混合，数据无限）
    - 真实退化模拟（模糊+下采样+噪声+JPEG）
    """

    def __init__(self, hr_dir, patch_size=128, scale=4, repeat=10):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.png')) +
                               glob.glob(os.path.join(hr_dir, '*.jpg')) +
                               glob.glob(os.path.join(hr_dir, '*.jpeg')) +
                               glob.glob(os.path.join(hr_dir, '*.bmp')))
        if len(self.hr_paths) == 0:
            raise RuntimeError(f'No images found in {hr_dir}')
        self.patch_size = patch_size
        self.scale = scale
        self.lr_size = patch_size // scale
        self.repeat = repeat
        self.synthesizer = ExposureSynthesizer()
        logger.info(f'Dataset: {len(self.hr_paths)} HR images, '
                    f'patch={patch_size}, scale=x{scale}, repeat={repeat}')

    def __len__(self):
        return len(self.hr_paths) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.hr_paths)
        hr = cv2.imread(self.hr_paths[idx], cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        # 随机裁剪HR patch
        hr = self._random_crop(hr)
        # 随机翻转旋转
        hr = self._random_augment(hr)
        # 曝光退化合成（渐变+普通+混合）
        degraded_hr, gain_map, exposure_type = self.synthesizer(hr)
        # 真实退化: 模糊 → 下采样 → 噪声
        lr = self._degrade(degraded_hr, self.scale)

        # 转tensor, 归一化到[0,1]
        hr_t = torch.from_numpy(hr.astype(np.float32).transpose(2, 0, 1)) / 255.0
        lr_t = torch.from_numpy(lr.astype(np.float32).transpose(2, 0, 1)) / 255.0
        gain_t = torch.from_numpy(gain_map).unsqueeze(0)  # [1, H, W]

        # exposure_type: 0=gradient, 1=uniform, 2=mixed
        type_map = {'gradient': 0, 'uniform': 1, 'mixed': 2}
        exp_type = torch.tensor(type_map[exposure_type], dtype=torch.long)

        return lr_t, hr_t, gain_t, exp_type

    def _random_crop(self, img):
        h, w = img.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            img = cv2.resize(img, (max(w, ps), max(h, ps)))
            h, w = img.shape[:2]
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        return img[y:y + ps, x:x + ps].copy()

    def _random_augment(self, img):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy()
        k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k).copy()
        return img

    def _degrade(self, img, scale):
        """真实退化: 随机模糊 → 下采样 → 随机噪声"""
        # 模糊
        if random.random() > 0.3:
            k = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            img = cv2.GaussianBlur(img, (k, k), sigma)
        # 下采样
        h, w = img.shape[:2]
        lr_h, lr_w = h // scale, w // scale
        method = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        img = cv2.resize(img, (lr_w, lr_h), interpolation=method)
        # 噪声
        if random.random() > 0.5:
            sigma = random.uniform(1, 10)
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img


# ====================================================================
# 第三部分: 损失函数
# ====================================================================

class SEFNetLoss(nn.Module):
    """
    多尺度曝光一致性损失（增强版，支持普通曝光+渐变曝光）

    总损失 = L1像素 + VGG感知 + 曝光一致性 + 亮度分布 + 曝光增益引导 + 曝光校准

    新增 _exposure_calibration_loss:
    - 普通曝光(uniform): 监督全局均值亮度一致（SR应把过曝/欠曝图恢复到正常亮度）
    - 渐变曝光(gradient): 监督空间亮度分布一致（SR应消除亮度渐变）
    - 混合曝光(mixed): 两者都监督
    """

    def __init__(self, vgg_net, l_pix=1.0, l_per=1.0, l_con=0.5, l_exp=0.3,
                 l_gain=0.1, l_cal=0.2):
        super().__init__()
        self.vgg = vgg_net
        self.l1 = nn.L1Loss()
        self.l_pix = l_pix
        self.l_per = l_per
        self.l_con = l_con
        self.l_exp = l_exp
        self.l_gain = l_gain
        self.l_cal = l_cal

    def _luminance(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def _brightness_dist_loss(self, sr, hr):
        bs = 16
        def stats(img):
            gray = self._luminance(img)
            B, C, H, W = gray.shape
            h2, w2 = H // bs, W // bs
            gray = gray[:, :, :h2 * bs, :w2 * bs]
            blocks = gray.unfold(2, bs, bs).unfold(3, bs, bs)
            return blocks.mean((-1, -2)), blocks.std((-1, -2))
        sr_m, sr_s = stats(sr)
        hr_m, hr_s = stats(hr)
        return self.l1(sr_m, hr_m) + self.l1(sr_s, hr_s)

    def _exposure_consistency(self, sr_feat, hr_feat, sr_lum, hr_lum):
        target_size = sr_feat.shape[2:]
        sr_e = F.adaptive_avg_pool2d(sr_lum, target_size)
        hr_e = F.adaptive_avg_pool2d(hr_lum, target_size)
        w = torch.sigmoid(5.0 - torch.abs(sr_e - hr_e) * 10)
        return torch.mean(torch.abs(sr_feat - hr_feat) * w)

    def _gain_guidance_loss(self, sr_lum, hr_lum, lr_lum):
        """曝光增益引导损失"""
        lr_lum_up = F.interpolate(
            lr_lum, size=hr_lum.shape[2:], mode='bilinear', align_corners=False
        )
        safe_lr = lr_lum_up.clamp(min=0.01)
        target_gain = (hr_lum / safe_lr).clamp(0.1, 10.0)
        sr_gain = (sr_lum / safe_lr).clamp(0.1, 10.0)
        mask = (torch.abs(target_gain - 1.0) > 0.1).float()
        if mask.sum() < 1:
            return torch.tensor(0.0, device=sr_lum.device)
        return self.l1(sr_gain * mask, target_gain * mask)

    def _exposure_calibration_loss(self, sr, hr, gain_map, exp_type):
        """
        曝光校准损失（核心新增）

        普通曝光 (type=1): 监督全局均值亮度一致
        渐变曝光 (type=0): 监督空间亮度分布一致
        混合曝光 (type=2): 两者都监督

        修复：原for循环+torch.tensor(0.0)初始化会断裂计算图。
        改用mask向量化：构造[B,2]权重矩阵，逐元素乘后求和。
        """
        sr_lum = self._luminance(sr)  # [B, 1, H, W]
        hr_lum = self._luminance(hr)

        # 全局均值校准（对普通曝光最重要）
        sr_mean = sr_lum.mean(dim=(2, 3))  # [B, 1]
        hr_mean = hr_lum.mean(dim=(2, 3))
        diff_mean = torch.abs(sr_mean - hr_mean)  # [B, 1]

        # 空间分布校准（对渐变曝光最重要）
        sr_std = sr_lum.std(dim=(2, 3))  # [B, 1]
        hr_std = hr_lum.std(dim=(2, 3))
        diff_std = torch.abs(sr_std - hr_std)  # [B, 1]

        # 向量化加权：根据exp_type构造权重
        # type=0(gradient): mean_w=0.5, std_w=2.0
        # type=1(uniform):  mean_w=2.0, std_w=0.5
        # type=2(mixed):    mean_w=1.0, std_w=1.0
        B = sr.shape[0]
        t = exp_type.float()  # [B]
        # w_mean: type0->0.5, type1->2.0, type2->1.0
        w_mean = torch.where(t == 0, 0.5, torch.where(t == 1, 2.0, 1.0)).unsqueeze(1)  # [B, 1]
        # w_std:  type0->2.0, type1->0.5, type2->1.0
        w_std = torch.where(t == 0, 2.0, torch.where(t == 1, 0.5, 1.0)).unsqueeze(1)   # [B, 1]

        total_cal = (w_mean * diff_mean + w_std * diff_std).mean()
        return total_cal

    def forward(self, sr, hr, lr=None, gain_map=None, exp_type=None):
        loss_pix = self.l1(sr, hr)
        sr_lum = self._luminance(sr)
        hr_lum = self._luminance(hr)

        if self.vgg is not None:
            sr_vgg = self.vgg(sr)
            hr_vgg = self.vgg(hr)
            loss_per = self.l1(sr_vgg, hr_vgg)
            loss_con = self._exposure_consistency(sr_vgg, hr_vgg, sr_lum, hr_lum)
        else:
            loss_per = torch.zeros((), device=sr.device)
            loss_con = torch.zeros((), device=sr.device)

        loss_exp = self._brightness_dist_loss(sr, hr)

        total = (self.l_pix * loss_pix + self.l_per * loss_per +
                 self.l_con * loss_con + self.l_exp * loss_exp)

        loss_dict = {'pix': loss_pix.item(), 'per': loss_per.item(),
                     'con': loss_con.item(), 'exp': loss_exp.item()}

        # 曝光增益引导损失
        if lr is not None:
            lr_lum = self._luminance(lr)
            loss_gain = self._gain_guidance_loss(sr_lum, hr_lum, lr_lum)
            total = total + self.l_gain * loss_gain
            loss_dict['gain'] = loss_gain.item()

        # 曝光校准损失（同时处理普通曝光和渐变曝光）
        if gain_map is not None and exp_type is not None:
            loss_cal = self._exposure_calibration_loss(sr, hr, gain_map, exp_type)
            total = total + self.l_cal * loss_cal
            loss_dict['cal'] = loss_cal.item()

        return total, loss_dict


def ra_d_loss(logits_real, logits_fake):
    """相对判别器损失"""
    lr = F.softplus(-(logits_real - logits_fake.mean())).mean()
    lf = F.softplus(logits_fake - logits_real.mean()).mean()
    return (lr + lf) / 2


def ra_g_loss(logits_real, logits_fake):
    """相对生成器损失"""
    lr = F.softplus(-(logits_real - logits_fake.mean())).mean()
    lf = F.softplus(logits_fake - logits_real.mean()).mean()
    return (lr + lf) / 2


# ====================================================================
# 第四部分: 工具函数
# ====================================================================

def save_ckpt(model, optimizer, epoch, loss, path, arch_cfg=None):
    # 统一 checkpoint 格式（见 utils.save_checkpoint）
    from utils import save_checkpoint
    save_checkpoint(model, path, optimizer=optimizer, epoch=epoch + 1,
                    loss=loss, arch_cfg=arch_cfg)


def load_ckpt(model, optimizer, path, device, arch_cfg=None):
    if not os.path.exists(path):
        return 0
    from utils import load_checkpoint
    state_dict, meta = load_checkpoint(path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and 'optimizer' in meta:
        try:
            optimizer.load_state_dict(meta['optimizer'])
        except Exception:
            logger.warning('optimizer state not restored from %s' % path)
    return meta.get('epoch', 0)


def save_samples(model, lr_imgs, save_path, device):
    model.eval()
    with torch.no_grad():
        sr = model(lr_imgs.to(device)).cpu().clamp_(0, 1)
    model.train()
    # 修复：nrow用ceil(sqrt)避免batch_size非完全平方数时丢图
    nrow = int(np.ceil(np.sqrt(lr_imgs.size(0))))
    save_image(sr, save_path, nrow=nrow)


# ====================================================================
# 第五部分: 三阶段训练
# ====================================================================

def stage1_pretrain(G, dataloader, sample_lr, device, cfg, args):
    """
    阶段1: 内容预热（base SR warmup）

    不再冻结随机初始化主干只训曝光模块——那样曝光分支学到的是对随机特征的补偿，
    而不是曝光先验。改为全网络 L1 + 曝光增益引导，先得到可用的基础超分能力；
    阶段2 再用曝光一致性/校准损失做精调。
    """
    save_dir = 'samples/stage1'
    ckpt_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info('Stage 1: Content Warmup (full network, L1 + gain guidance)')
    logger.info('=' * 60)

    for p in G.parameters():
        p.requires_grad = True

    trainable = [p for p in G.parameters() if p.requires_grad]
    logger.info(f'Trainable params: {sum(p.numel() for p in trainable):,}')

    # 曝光模块稍高 lr，主干略低，保证曝光分支仍能快速跟上
    optimizer = torch.optim.Adam([
        {'params': G.exposure_stats.parameters(), 'lr': 1e-4},
        {'params': G.relative_encoder.parameters(), 'lr': 1e-4},
        {'params': G.conv_first.parameters(), 'lr': 5e-5},
        {'params': G.rrdb_blocks.parameters(), 'lr': 5e-5},
        {'params': G.LR_conv.parameters(), 'lr': 5e-5},
        {'params': G.upsample.parameters(), 'lr': 5e-5},
        {'params': G.HR_conv0.parameters(), 'lr': 5e-5},
        {'params': G.HR_conv1.parameters(), 'lr': 5e-5},
    ])
    ckpt_path = os.path.join(ckpt_dir, 'g_stage1.pth')
    start_ep = load_ckpt(G, optimizer, ckpt_path, device)

    G.train()
    for ep in range(start_ep, args.stage1_epochs):
        ep_loss, n = 0, 0
        t0 = time.time()
        for lr_imgs, hr_imgs, gain_map, exp_type in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            gain_map = gain_map.to(device)
            exp_type = exp_type.to(device)

            sr = G(lr_imgs)
            loss = F.l1_loss(sr, hr_imgs)

            # 曝光增益引导
            sr_lum = 0.299 * sr[:, 0:1] + 0.587 * sr[:, 1:2] + 0.114 * sr[:, 2:3]
            hr_lum = 0.299 * hr_imgs[:, 0:1] + 0.587 * hr_imgs[:, 1:2] + 0.114 * hr_imgs[:, 2:3]
            lr_lum = 0.299 * lr_imgs[:, 0:1] + 0.587 * lr_imgs[:, 1:2] + 0.114 * lr_imgs[:, 2:3]
            lr_lum_hr = F.interpolate(lr_lum, size=hr_lum.shape[2:], mode='bilinear', align_corners=False)
            safe_lr = lr_lum_hr.clamp(min=0.01)
            target_gain = (hr_lum / safe_lr).clamp(0.1, 10.0)
            sr_gain = (sr_lum / safe_lr).clamp(0.1, 10.0)
            gain_mask = (torch.abs(target_gain - 1.0) > 0.1).float()
            if gain_mask.sum() > 0:
                loss = loss + 0.1 * F.l1_loss(sr_gain * gain_mask, target_gain * gain_mask)

            # 曝光校准损失（向量化，不破坏计算图）
            sr_mean = sr_lum.mean(dim=(2, 3))  # [B, 1]
            hr_mean = hr_lum.mean(dim=(2, 3))
            sr_std = sr_lum.std(dim=(2, 3))
            hr_std = hr_lum.std(dim=(2, 3))
            diff_mean = torch.abs(sr_mean - hr_mean)
            diff_std = torch.abs(sr_std - hr_std)
            t = exp_type.float().unsqueeze(1)
            w_mean = torch.where(t == 0, 0.5, torch.where(t == 1, 2.0, 1.0))
            w_std = torch.where(t == 0, 2.0, torch.where(t == 1, 0.5, 1.0))
            loss_cal = (w_mean * diff_mean + w_std * diff_std).mean()
            loss = loss + 0.2 * loss_cal

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            n += 1

        avg = ep_loss / max(n, 1)
        logger.info(f'[Stage1] Epoch [{ep}/{args.stage1_epochs}] '
                     f'time: {time.time()-t0:.1f}s  loss: {avg:.6f}')

        if (ep + 1) % 10 == 0:
            save_samples(G, sample_lr, os.path.join(save_dir, f'train_{ep}.png'), device)
            save_ckpt(G, optimizer, ep, avg, ckpt_path,
                      arch_cfg={'nf': args.nf, 'nb': args.nb, 'gc': args.gc,
                                'upscale': args.scale, 'context_dim': 64,
                                'exposure_feat_dim': 32, 'upsample_mode': 'pixelshuffle'})
            logger.info(f'[*] saved checkpoint + samples')


def stage2_joint(G, vgg_net, dataloader, sample_lr, device, cfg, args):
    """
    阶段2: 联合训练，曝光模块更高学习率
    支持普通曝光+渐变曝光+混合曝光
    """
    save_dir = 'samples/stage2'
    ckpt_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info('Stage 2: Joint Training (gradient + uniform + mixed exposure)')
    logger.info('=' * 60)

    s1_path = os.path.join(ckpt_dir, 'g_stage1.pth')
    if os.path.exists(s1_path):
        load_ckpt(G, None, s1_path, device)
        logger.info('Loaded stage1 weights')

    for p in G.parameters():
        p.requires_grad = True

    criterion = SEFNetLoss(vgg_net,
                           l_pix=cfg.TRAIN.lambda_pixel,
                           l_per=cfg.TRAIN.lambda_percep,
                           l_con=cfg.TRAIN.lambda_consistency,
                           l_exp=cfg.TRAIN.lambda_exposure,
                           l_gain=cfg.TRAIN.lambda_gan * 100,
                           l_cal=0.2).to(device)

    lr_exp = cfg.TRAIN.lr_init
    lr_main = lr_exp * 0.5
    optimizer = torch.optim.Adam([
        {'params': G.exposure_stats.parameters(), 'lr': lr_exp},
        {'params': G.relative_encoder.parameters(), 'lr': lr_exp},
        {'params': G.conv_first.parameters(), 'lr': lr_main},
        {'params': G.rrdb_blocks.parameters(), 'lr': lr_main},
        {'params': G.LR_conv.parameters(), 'lr': lr_main},
        {'params': G.upsample.parameters(), 'lr': lr_main},
        {'params': G.HR_conv0.parameters(), 'lr': lr_main},
        {'params': G.HR_conv1.parameters(), 'lr': lr_main},
    ])

    ckpt_path = os.path.join(ckpt_dir, 'g_stage2.pth')
    start_ep = load_ckpt(G, optimizer, ckpt_path, device)

    G.train()
    for ep in range(start_ep, args.stage2_epochs):
        if ep > 0 and ep % args.decay_every == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            logger.info(f'** lr decayed to {optimizer.param_groups[0]["lr"]:.2e}')

        ep_loss, n = 0, 0
        t0 = time.time()
        for lr_imgs, hr_imgs, gain_map, exp_type in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            gain_map = gain_map.to(device)
            exp_type = exp_type.to(device)

            sr = G(lr_imgs)
            loss, ld = criterion(sr, hr_imgs, lr=lr_imgs,
                                 gain_map=gain_map, exp_type=exp_type)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            n += 1

        avg = ep_loss / max(n, 1)
        logger.info(f'[Stage2] Epoch [{ep}/{args.stage2_epochs}] '
                     f'time: {time.time()-t0:.1f}s  loss: {avg:.6f}')

        if (ep + 1) % 10 == 0:
            save_samples(G, sample_lr, os.path.join(save_dir, f'train_{ep}.png'), device)
            save_ckpt(G, optimizer, ep, avg, ckpt_path,
                      arch_cfg={'nf': args.nf, 'nb': args.nb, 'gc': args.gc,
                                'upscale': args.scale, 'context_dim': 64,
                                'exposure_feat_dim': 32, 'upsample_mode': 'pixelshuffle'})
            logger.info(f'[*] saved checkpoint + samples')


def stage3_gan(G, D, vgg_net, dataloader, sample_lr, device, cfg, args):
    """
    阶段3: GAN对抗训练 (RaGAN)
    修复：
    1. 生成器和判别器都添加梯度裁剪（GAN训练稳定性关键）
    2. 传入lr_imgs启用曝光增益引导损失
    """
    save_dir = 'samples/stage3'
    ckpt_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info('Stage 3: GAN Adversarial Training (RaGAN)')
    logger.info('=' * 60)

    s2_path = os.path.join(ckpt_dir, 'g_stage2.pth')
    if os.path.exists(s2_path):
        load_ckpt(G, None, s2_path, device)
        logger.info('Loaded stage2 weights')

    G.train()
    D.to(device).train()

    criterion = SEFNetLoss(vgg_net,
                           l_pix=cfg.TRAIN.lambda_pixel,
                           l_per=cfg.TRAIN.lambda_percep,
                           l_con=cfg.TRAIN.lambda_consistency,
                           l_exp=cfg.TRAIN.lambda_exposure,
                           l_gain=cfg.TRAIN.lambda_gan * 100,
                           l_cal=0.2).to(device)

    lr_gan = cfg.TRAIN.stage3_lr if hasattr(cfg.TRAIN, 'stage3_lr') else 1e-4
    g_opt = torch.optim.Adam(G.parameters(), lr=lr_gan)
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_gan)

    g_ckpt = os.path.join(ckpt_dir, 'g_stage3.pth')
    d_ckpt = os.path.join(ckpt_dir, 'd_stage3.pth')
    start_ep = load_ckpt(G, g_opt, g_ckpt, device)
    load_ckpt(D, d_opt, d_ckpt, device)

    for ep in range(start_ep, args.stage3_epochs):
        if ep > 0 and ep % args.decay_every == 0:
            for pg in g_opt.param_groups + d_opt.param_groups:
                pg['lr'] *= 0.5

        ep_g, ep_d, n = 0, 0, 0
        t0 = time.time()
        for lr_imgs, hr_imgs, gain_map, exp_type in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            gain_map = gain_map.to(device)
            exp_type = exp_type.to(device)

            # update D
            sr_det = G(lr_imgs).detach()
            d_loss = ra_d_loss(D(hr_imgs), D(sr_det))
            d_opt.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            d_opt.step()

            # update G
            sr = G(lr_imgs)
            content, ld = criterion(sr, hr_imgs, lr=lr_imgs,
                                    gain_map=gain_map, exp_type=exp_type)
            gan = ra_g_loss(D(hr_imgs), D(sr))
            g_loss = content + cfg.TRAIN.lambda_gan * gan
            g_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            g_opt.step()

            ep_g += g_loss.item()
            ep_d += d_loss.item()
            n += 1

        logger.info(f'[Stage3] Epoch [{ep}/{args.stage3_epochs}] '
                     f'time: {time.time()-t0:.1f}s  G: {ep_g/n:.6f}  D: {ep_d/n:.6f}')

        if (ep + 1) % 10 == 0:
            save_samples(G, sample_lr, os.path.join(save_dir, f'train_{ep}.png'), device)
            save_ckpt(G, g_opt, ep, ep_g / n, g_ckpt,
                      arch_cfg={'nf': args.nf, 'nb': args.nb, 'gc': args.gc,
                                'upscale': args.scale, 'context_dim': 64,
                                'exposure_feat_dim': 32, 'upsample_mode': 'pixelshuffle'})
            save_ckpt(D, d_opt, ep, ep_d / n, d_ckpt)
            logger.info(f'[*] saved checkpoint + samples')


# ====================================================================
# 主函数
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='SEFNet Training')
    parser.add_argument('--hr_dir', type=str, default='./train_images',
                        help='HR image directory (any folder of images)')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--stage1_epochs', type=int, default=50)
    parser.add_argument('--stage2_epochs', type=int, default=200)
    parser.add_argument('--stage3_epochs', type=int, default=100)
    parser.add_argument('--decay_every', type=int, default=50)
    parser.add_argument('--stage', type=str, default='all', choices=['1', '2', '3', 'all'])
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16, help='number of RRDB blocks')
    parser.add_argument('--gc', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f'Device: {device}')

    # 目录
    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    # 文件日志
    fh = logging.FileHandler(f'log/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    # ===== 数据 =====
    dataset = SEFNetTrainDataset(args.hr_dir, args.patch_size, args.scale, repeat=50)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f'Dataloader: {len(dataloader)} iters/epoch')

    # 固定样张（训练期间监控用）
    # 修复：__getitem__现在返回4个值，需要正确解包
    sample_lr = dataset[0][0].unsqueeze(0)  # 只取lr_t
    for i in range(1, min(args.batch_size, 4)):
        sample_lr = torch.cat([sample_lr, dataset[i][0].unsqueeze(0)], dim=0)
    save_image(sample_lr, 'samples/_sample_lr.png', nrow=2)
    logger.info(f'Sample LR shape: {sample_lr.shape}')

    # ===== 模型 =====
    G = arch.RRDB_Net(3, 3, args.nf, args.nb, gc=args.gc, upscale=args.scale,
                      norm_type=None, act_type='leakyrelu', mode='CNA',
                      upsample_mode='pixelshuffle',
                      context_dim=64, exposure_feat_dim=32).to(device)
    logger.info(f'Generator params: {sum(p.numel() for p in G.parameters()):,}')

    D = arch.Discriminator_VGG(in_nc=3, nf=64)
    logger.info(f'Discriminator params: {sum(p.numel() for p in D.parameters()):,}')

    # VGG 仅 stage2/3 需要；stage1 可完全离线
    need_vgg = args.stage in ('2', '3', 'all')
    vgg_net = None
    if need_vgg:
        vgg_net = arch.VGGFeatureExtractor(
            feature_layer=34,
            weights_path=os.path.join('weights', 'vgg19-dcbb9e9d.pth'),
        ).to(device)
        vgg_net.eval()
    else:
        logger.info('Stage1 only: skip VGG load (offline-friendly)')

    # ===== 训练 =====
    if args.stage == '1' or args.stage == 'all':
        stage1_pretrain(G, dataloader, sample_lr, device, config, args)
    if args.stage == '2' or args.stage == 'all':
        stage2_joint(G, vgg_net, dataloader, sample_lr, device, config, args)
    if args.stage == '3' or args.stage == 'all':
        stage3_gan(G, D, vgg_net, dataloader, sample_lr, device, config, args)

    # 保存最终模型（统一格式，含 arch 元信息，test.py 可自动建网）
    final_path = 'checkpoint/SEFNet_G_final.pth'
    arch_cfg = {
        'nf': args.nf, 'nb': args.nb, 'gc': args.gc, 'upscale': args.scale,
        'context_dim': 64, 'exposure_feat_dim': 32, 'upsample_mode': 'pixelshuffle',
    }
    save_checkpoint(G, final_path, epoch=args.stage1_epochs + args.stage2_epochs + args.stage3_epochs,
                    arch_cfg=arch_cfg)
    logger.info(f'Final model saved: {final_path}')
    logger.info('Training complete!')


if __name__ == '__main__':
    main()

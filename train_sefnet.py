#! /usr/bin/python
# -*- coding: utf8 -*-
"""
SEFNet Training Script - 可直接运行的完整训练脚本

功能:
  1. 自动从任意图片目录生成曝光渐变训练数据
  2. 三阶段训练: 曝光模块预训练 → 联合训练 → GAN对抗训练
  3. 自动保存样张、checkpoint、训练日志

使用方式:
  # 用测试图作为演示（小数据集，快速验证流程）
  python train_sefnet.py --hr_dir ./test_images --epochs 5

  # 用自己的图片目录（推荐放100+张图）
  python train_sefnet.py --hr_dir /path/to/images --epochs 100

  # 用DIV2K数据集（正式训练）
  python train_sefnet.py --hr_dir data/DIV2K_train_HR --epochs 500
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

# ===== 日志 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('SEFNet')


# ====================================================================
# 第一部分: 曝光渐变数据合成（训练时在线生成，不需要预处理）
# ====================================================================

class ExposureGradientSynthesizer:
    """
    在线曝光渐变合成器
    每次调用随机生成不同方向和强度的曝光渐变
    """

    def __init__(self, gain_range=(0.2, 3.0), blur_range=(51, 201)):
        self.gain_range = gain_range
        self.blur_range = blur_range
        # 六种渐变类型及其采样权重
        self.types = ['vertical', 'horizontal', 'radial_center',
                      'radial_edge', 'diagonal', 'irregular']
        self.weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]

    def __call__(self, img):
        """
        img: numpy array [H, W, 3], uint8, RGB
        返回: (degraded, gain_map)
            degraded: [H, W, 3] uint8
            gain_map: [H, W] float32
        """
        h, w = img.shape[:2]
        grad_type = random.choices(self.types, weights=self.weights, k=1)[0]
        intensity = random.uniform(self.gain_range[0], self.gain_range[1])
        reverse = random.random() > 0.5

        gain_map = self._make_gain(h, w, grad_type, intensity, reverse)
        # 平滑
        k = random.choice(range(self.blur_range[0], self.blur_range[1] + 1, 2))
        gain_map = cv2.GaussianBlur(gain_map, (k, k), 0)

        img_f = img.astype(np.float32)
        degraded = np.clip(img_f * gain_map[:, :, np.newaxis], 0, 255).astype(np.uint8)
        return degraded, gain_map.astype(np.float32)

    def _make_gain(self, h, w, grad_type, intensity, reverse):
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
    SEFNet 训练数据集

    数据流:
    HR图 → 随机裁剪 → 随机翻转旋转 → 曝光渐变合成 → 真实退化下采样 → LR输入
                                     → 正常曝光HR → 监督目标(GT)

    支持:
    - 任意图片目录作为HR源
    - 在线生成曝光渐变（每次不同，数据无限）
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
        self.synthesizer = ExposureGradientSynthesizer()
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
        # 曝光渐变合成
        degraded_hr, gain_map = self.synthesizer(hr)
        # 真实退化: 模糊 → 下采样 → 噪声
        lr = self._degrade(degraded_hr, self.scale)

        # 转tensor, 归一化到[0,1]
        hr_t = torch.from_numpy(hr.astype(np.float32).transpose(2, 0, 1)) / 255.0
        lr_t = torch.from_numpy(lr.astype(np.float32).transpose(2, 0, 1)) / 255.0
        return lr_t, hr_t

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
    多尺度曝光一致性损失
    = L1像素损失 + VGG感知损失 + 曝光一致性损失 + 亮度分布损失
    """

    def __init__(self, vgg_net, l_pix=1.0, l_per=1.0, l_con=0.5, l_exp=0.3):
        super().__init__()
        self.vgg = vgg_net
        self.l1 = nn.L1Loss()
        self.l_pix = l_pix
        self.l_per = l_per
        self.l_con = l_con
        self.l_exp = l_exp

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
        w = torch.exp(-torch.abs(sr_e - hr_e) * 5)
        return torch.mean(torch.abs(sr_feat - hr_feat) * w)

    def forward(self, sr, hr):
        loss_pix = self.l1(sr, hr)
        sr_vgg = self.vgg(sr)
        hr_vgg = self.vgg(hr)
        loss_per = self.l1(sr_vgg, hr_vgg)
        sr_lum = self._luminance(sr)
        hr_lum = self._luminance(hr)
        loss_con = self._exposure_consistency(sr_vgg, hr_vgg, sr_lum, hr_lum)
        loss_exp = self._brightness_dist_loss(sr, hr)
        total = (self.l_pix * loss_pix + self.l_per * loss_per +
                 self.l_con * loss_con + self.l_exp * loss_exp)
        return total, {'pix': loss_pix.item(), 'per': loss_per.item(),
                        'con': loss_con.item(), 'exp': loss_exp.item()}


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

def save_ckpt(model, optimizer, epoch, loss, path):
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'loss': loss}, path)


def load_ckpt(model, optimizer, path, device):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        if optimizer and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)
    return 0


def save_samples(model, lr_imgs, save_path, device):
    model.eval()
    with torch.no_grad():
        sr = model(lr_imgs.to(device)).cpu().clamp_(0, 1)
    model.train()
    save_image(sr, save_path, nrow=int(np.sqrt(lr_imgs.size(0))))


# ====================================================================
# 第五部分: 三阶段训练
# ====================================================================

def stage1_pretrain(G, dataloader, sample_lr, device, cfg, args):
    """阶段1: 冻结主网络，只训练曝光模块"""
    save_dir = 'samples/stage1'
    ckpt_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info('Stage 1: Exposure Module Pre-training')
    logger.info('=' * 60)

    for name, p in G.named_parameters():
        p.requires_grad = ('exposure_stats' in name or 'relative_encoder' in name)

    trainable = [p for p in G.parameters() if p.requires_grad]
    logger.info(f'Trainable params: {sum(p.numel() for p in trainable):,}')

    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    ckpt_path = os.path.join(ckpt_dir, 'g_stage1.pth')
    start_ep = load_ckpt(G, optimizer, ckpt_path, device)

    G.train()
    for ep in range(start_ep, args.stage1_epochs):
        ep_loss, n = 0, 0
        t0 = time.time()
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr = G(lr_imgs)
            loss = F.l1_loss(sr, hr_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n += 1

        avg = ep_loss / max(n, 1)
        logger.info(f'[Stage1] Epoch [{ep}/{args.stage1_epochs}] '
                     f'time: {time.time()-t0:.1f}s  loss: {avg:.6f}')

        if (ep + 1) % 10 == 0:
            save_samples(G, sample_lr, os.path.join(save_dir, f'train_{ep}.png'), device)
            save_ckpt(G, optimizer, ep, avg, ckpt_path)
            logger.info(f'[*] saved checkpoint + samples')


def stage2_joint(G, vgg_net, dataloader, sample_lr, device, cfg, args):
    """阶段2: 联合训练，曝光模块更高学习率"""
    save_dir = 'samples/stage2'
    ckpt_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info('Stage 2: Joint Training with Exposure Consistency Loss')
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
                           l_exp=cfg.TRAIN.lambda_exposure).to(device)

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
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr = G(lr_imgs)
            loss, ld = criterion(sr, hr_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n += 1

        avg = ep_loss / max(n, 1)
        logger.info(f'[Stage2] Epoch [{ep}/{args.stage2_epochs}] '
                     f'time: {time.time()-t0:.1f}s  loss: {avg:.6f}')

        if (ep + 1) % 10 == 0:
            save_samples(G, sample_lr, os.path.join(save_dir, f'train_{ep}.png'), device)
            save_ckpt(G, optimizer, ep, avg, ckpt_path)
            logger.info(f'[*] saved checkpoint + samples')


def stage3_gan(G, D, vgg_net, dataloader, sample_lr, device, cfg, args):
    """阶段3: GAN对抗训练 (RaGAN)"""
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
                           l_exp=cfg.TRAIN.lambda_exposure).to(device)

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
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # update D
            sr_det = G(lr_imgs).detach()
            d_loss = ra_d_loss(D(hr_imgs), D(sr_det))
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # update G
            sr = G(lr_imgs)
            content, ld = criterion(sr, hr_imgs)
            gan = ra_g_loss(D(hr_imgs), D(sr))
            g_loss = content + cfg.TRAIN.lambda_gan * gan
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            ep_g += g_loss.item()
            ep_d += d_loss.item()
            n += 1

        logger.info(f'[Stage3] Epoch [{ep}/{args.stage3_epochs}] '
                     f'time: {time.time()-t0:.1f}s  G: {ep_g/n:.6f}  D: {ep_d/n:.6f}')

        if (ep + 1) % 10 == 0:
            save_samples(G, sample_lr, os.path.join(save_dir, f'train_{ep}.png'), device)
            save_ckpt(G, g_opt, ep, ep_g / n, g_ckpt)
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
    lr_sample, hr_sample = dataset[0]
    sample_lr = lr_sample.unsqueeze(0)
    for i in range(1, min(args.batch_size, 4)):
        lr_s, _ = dataset[i]
        sample_lr = torch.cat([sample_lr, lr_s.unsqueeze(0)], dim=0)
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

    vgg_net = arch.VGGFeatureExtractor(feature_layer=34).to(device)
    vgg_net.eval()

    # ===== 训练 =====
    if args.stage == '1' or args.stage == 'all':
        stage1_pretrain(G, dataloader, sample_lr, device, config, args)
    if args.stage == '2' or args.stage == 'all':
        stage2_joint(G, vgg_net, dataloader, sample_lr, device, config, args)
    if args.stage == '3' or args.stage == 'all':
        stage3_gan(G, D, vgg_net, dataloader, sample_lr, device, config, args)

    # 保存最终模型
    final_path = 'checkpoint/SEFNet_G_final.pth'
    torch.save(G.state_dict(), final_path)
    logger.info(f'Final model saved: {final_path}')
    logger.info('Training complete!')


if __name__ == '__main__':
    main()

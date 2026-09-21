#! /usr/bin/python
# -*- coding: utf8 -*-
"""
SEFNet Training Script
三阶段训练策略

阶段1: 冻结主网络，只训练曝光相关模块（曝光模块预训练）
阶段2: 联合训练，曝光模块学习率更高（曝光一致性损失）
阶段3: 加入GAN对抗训练，微调整体（RaGAN）

Usage:
    python train.py --stage 1        # 阶段1：曝光模块预训练
    python train.py --stage 2        # 阶段2：联合训练
    python train.py --stage 3        # 阶段3：GAN对抗训练
    python train.py --stage all      # 全流程训练
"""

import os, time, random
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import config
import architecture as arch
from block import (MultiScaleExposureConsistencyLoss, compute_luminance,
                   relativistic_discriminator_loss, relativistic_generator_loss)
from data import SRDataset, ExposureGradientSynthesizer

###====================== 日志设置 ===========================###
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt.get('epoch', 0)
    return 0


def save_sample_images(model, sample_lr, save_dir, epoch, device, ni):
    """保存样张图片"""
    model.eval()
    with torch.no_grad():
        out = model(sample_lr.to(device))
    model.train()
    out = out.cpu().clamp_(0, 1)
    save_image(out, os.path.join(save_dir, 'train_%d.png' % epoch), nrow=ni)


###====================== HYPER-PARAMETERS ===========================###
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
ni = int(np.sqrt(batch_size))


def train_stage1(generator, dataloader, sample_lr, device, cfg):
    """
    阶段1: 曝光模块预训练
    只训练 GlobalExposureStatistics 和 RelativeExposureEncoder
    冻结 RRDB块和上采样
    曝光模块预训练阶段
    """
    save_dir = "samples/stage1_ginit"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Stage 1: Exposure Module Pre-training (G init)")
    logger.info("=" * 60)

    # 冻结主网络参数
    for name, param in generator.named_parameters():
        if 'exposure_stats' not in name and 'relative_encoder' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = [p for n, p in generator.named_parameters() if p.requires_grad]
    logger.info("Trainable params (stage1): {:,}".format(sum(p.numel() for p in trainable_params)))

    optimizer = torch.optim.Adam(trainable_params, lr=cfg.TRAIN.stage1_lr,
                                 betas=(cfg.TRAIN.beta1, cfg.TRAIN.beta2))

    start_epoch = 0
    ckpt_path = os.path.join(checkpoint_dir, 'g_sefnet_stage1.pth')
    if os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(generator, optimizer, ckpt_path, device)
        logger.info("Loaded checkpoint: {} (epoch {})".format(ckpt_path, start_epoch))

    # 固定学习率
    logger.info(" ** fixed learning rate: %f (for init G)" % cfg.TRAIN.stage1_lr)

    generator.train()
    for epoch in range(start_epoch, cfg.TRAIN.stage1_epochs):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        for idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            step_time = time.time()
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_output = generator(lr_imgs)
            loss = F.l1_loss(sr_output, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse_loss += loss.item()
            n_iter += 1

            if (idx + 1) % cfg.TRAIN.log_interval == 0:
                logger.info("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f" %
                            (epoch, cfg.TRAIN.stage1_epochs, n_iter,
                             time.time() - step_time, loss.item()))

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % \
              (epoch, cfg.TRAIN.stage1_epochs, time.time() - epoch_time, total_mse_loss / n_iter)
        logger.info(log)

        # 快速评估并保存样张
        if epoch != 0 and epoch % 10 == 0:
            save_sample_images(generator, sample_lr, save_dir, epoch, device, ni)
            logger.info("[*] save images")

        # 保存模型
        if epoch != 0 and epoch % 10 == 0:
            save_checkpoint(generator, optimizer, epoch, total_mse_loss / n_iter, ckpt_path)
            logger.info("[*] save checkpoint")


def train_stage2(generator, vgg_net, dataloader, sample_lr, device, cfg):
    """
    阶段2: 联合训练
    所有参数都参与训练，曝光模块使用更高学习率
    使用多尺度曝光一致性损失
    """
    save_dir = "samples/stage2_joint"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = "checkpoint"

    logger.info("=" * 60)
    logger.info("Stage 2: Joint Training with Exposure Consistency Loss")
    logger.info("=" * 60)

    # 加载阶段1权重
    stage1_path = os.path.join(checkpoint_dir, 'g_sefnet_stage1.pth')
    if os.path.exists(stage1_path):
        load_checkpoint(generator, None, stage1_path, device)
        logger.info("Loaded stage1 weights from %s" % stage1_path)

    # 解冻所有参数
    for param in generator.parameters():
        param.requires_grad = True

    logger.info("Total params: {:,}".format(sum(p.numel() for p in generator.parameters())))

    criterion = MultiScaleExposureConsistencyLoss(
        vgg_net,
        lambda_pixel=cfg.TRAIN.lambda_pixel,
        lambda_percep=cfg.TRAIN.lambda_percep,
        lambda_consistency=cfg.TRAIN.lambda_consistency,
        lambda_exposure=cfg.TRAIN.lambda_exposure
    ).to(device)

    # 差异化学习率（曝光模块学习率更高）
    optimizer = torch.optim.Adam([
        {'params': generator.exposure_stats.parameters(), 'lr': cfg.TRAIN.stage2_lr},
        {'params': generator.relative_encoder.parameters(), 'lr': cfg.TRAIN.stage2_lr},
        {'params': generator.conv_first.parameters(), 'lr': cfg.TRAIN.stage2_lr * 0.5},
        {'params': generator.rrdb_blocks.parameters(), 'lr': cfg.TRAIN.stage2_lr * 0.5},
        {'params': generator.LR_conv.parameters(), 'lr': cfg.TRAIN.stage2_lr * 0.5},
        {'params': generator.upsample.parameters(), 'lr': cfg.TRAIN.stage2_lr * 0.5},
        {'params': generator.HR_conv0.parameters(), 'lr': cfg.TRAIN.stage2_lr * 0.5},
        {'params': generator.HR_conv1.parameters(), 'lr': cfg.TRAIN.stage2_lr * 0.5},
    ], betas=(cfg.TRAIN.beta1, cfg.TRAIN.beta2))

    start_epoch = 0
    ckpt_path = os.path.join(checkpoint_dir, 'g_sefnet_stage2.pth')
    if os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(generator, optimizer, ckpt_path, device)
        logger.info("Loaded checkpoint: {} (epoch {})".format(ckpt_path, start_epoch))

    logger.info(" ** init lr: %f  decay_every: %d, lr_decay: %f" %
                (cfg.TRAIN.stage2_lr, cfg.TRAIN.decay_every, cfg.TRAIN.lr_decay))

    generator.train()
    for epoch in range(start_epoch, cfg.TRAIN.stage2_epochs):
        # 学习率衰减
        if epoch != 0 and epoch % cfg.TRAIN.decay_every == 0:
            new_lr = cfg.TRAIN.stage2_lr * (cfg.TRAIN.lr_decay ** (epoch // cfg.TRAIN.decay_every))
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(new_lr * 0.5, 1e-6) if param_group['lr'] < cfg.TRAIN.stage2_lr else new_lr
            logger.info(" ** new learning rate: %f" % new_lr)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        for idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            step_time = time.time()
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_output = generator(lr_imgs)
            sr_lum = compute_luminance(sr_output)
            hr_lum = compute_luminance(hr_imgs)

            loss, loss_dict = criterion(sr_output, hr_imgs, sr_lum, hr_lum)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_iter += 1

            if (idx + 1) % cfg.TRAIN.log_interval == 0:
                logger.info("Epoch [%2d/%2d] %4d time: %4.4fs, loss: %.8f "
                            "(mse: %.6f vgg: %.6f cons: %.6f exp: %.6f)" %
                            (epoch, cfg.TRAIN.stage2_epochs, n_iter,
                             time.time() - step_time, loss.item(),
                             loss_dict['pixel'], loss_dict['percep'],
                             loss_dict['consistency'], loss_dict['exposure']))

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, loss: %.8f" % \
              (epoch, cfg.TRAIN.stage2_epochs, time.time() - epoch_time, total_loss / n_iter)
        logger.info(log)

        if epoch != 0 and epoch % 10 == 0:
            save_sample_images(generator, sample_lr, save_dir, epoch, device, ni)
            logger.info("[*] save images")

        if epoch != 0 and epoch % 10 == 0:
            save_checkpoint(generator, optimizer, epoch, total_loss / n_iter, ckpt_path)
            logger.info("[*] save checkpoint")


def train_stage3(generator, discriminator, vgg_net, dataloader, sample_lr, device, cfg):
    """
    阶段3: GAN对抗训练
    加入判别器，使用相对判别器损失
    """
    save_dir = "samples/stage3_gan"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = "checkpoint"

    logger.info("=" * 60)
    logger.info("Stage 3: GAN Adversarial Training (RaGAN)")
    logger.info("=" * 60)

    # 加载阶段2权重
    stage2_path = os.path.join(checkpoint_dir, 'g_sefnet_stage2.pth')
    if os.path.exists(stage2_path):
        load_checkpoint(generator, None, stage2_path, device)
        logger.info("Loaded stage2 weights from %s" % stage2_path)

    discriminator.to(device)

    generator.print_params = lambda verbose=False: None  # 兼容

    criterion = MultiScaleExposureConsistencyLoss(
        vgg_net,
        lambda_pixel=cfg.TRAIN.lambda_pixel,
        lambda_percep=cfg.TRAIN.lambda_percep,
        lambda_consistency=cfg.TRAIN.lambda_consistency,
        lambda_exposure=cfg.TRAIN.lambda_exposure
    ).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.TRAIN.stage3_lr,
                                   betas=(cfg.TRAIN.beta1, cfg.TRAIN.beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.TRAIN.stage3_lr,
                                   betas=(cfg.TRAIN.beta1, cfg.TRAIN.beta2))

    start_epoch = 0
    g_ckpt = os.path.join(checkpoint_dir, 'g_sefnet_stage3.pth')
    d_ckpt = os.path.join(checkpoint_dir, 'd_sefnet_stage3.pth')
    if os.path.exists(g_ckpt):
        start_epoch = load_checkpoint(generator, g_optimizer, g_ckpt, device)
    if os.path.exists(d_ckpt):
        load_checkpoint(discriminator, d_optimizer, d_ckpt, device)

    generator.train()
    discriminator.train()

    for epoch in range(start_epoch, cfg.TRAIN.stage3_epochs):
        # 学习率衰减
        if epoch != 0 and epoch % cfg.TRAIN.decay_every == 0:
            new_lr = cfg.TRAIN.stage3_lr * (cfg.TRAIN.lr_decay ** (epoch // cfg.TRAIN.decay_every))
            for pg in g_optimizer.param_groups:
                pg['lr'] = new_lr
            for pg in d_optimizer.param_groups:
                pg['lr'] = new_lr
            logger.info(" ** new learning rate: %f" % new_lr)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        for idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            step_time = time.time()
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # ===== 训练判别器 =====
            sr_output = generator(lr_imgs).detach()
            logits_real = discriminator(hr_imgs)
            logits_fake = discriminator(sr_output)
            d_loss = relativistic_discriminator_loss(logits_real, logits_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ===== 训练生成器 =====
            sr_output = generator(lr_imgs)
            logits_real = discriminator(hr_imgs)
            logits_fake = discriminator(sr_output)

            sr_lum = compute_luminance(sr_output)
            hr_lum = compute_luminance(hr_imgs)
            content_loss, loss_dict = criterion(sr_output, hr_imgs, sr_lum, hr_lum)
            gan_loss = relativistic_generator_loss(logits_real, logits_fake)
            g_loss = content_loss + cfg.TRAIN.lambda_gan * gan_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            n_iter += 1

            if (idx + 1) % cfg.TRAIN.log_interval == 0:
                logger.info("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f "
                            "(mse: %.6f vgg: %.6f adv: %.6f)" %
                            (epoch, cfg.TRAIN.stage3_epochs, n_iter,
                             time.time() - step_time, d_loss.item(), g_loss.item(),
                             loss_dict['pixel'], loss_dict['percep'], gan_loss.item()))

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % \
              (epoch, cfg.TRAIN.stage3_epochs, time.time() - epoch_time,
               total_d_loss / n_iter, total_g_loss / n_iter)
        logger.info(log)

        if epoch != 0 and epoch % 10 == 0:
            save_sample_images(generator, sample_lr, save_dir, epoch, device, ni)
            logger.info("[*] save images")

        if epoch != 0 and epoch % 10 == 0:
            save_checkpoint(generator, g_optimizer, epoch, total_g_loss / n_iter, g_ckpt)
            save_checkpoint(discriminator, d_optimizer, epoch, total_d_loss / n_iter, d_ckpt)
            logger.info("[*] save checkpoint")


def evaluate(model, valid_hr_path, valid_lr_path, device, cfg):
    """
    评估函数，在验证集上计算PSNR/SSIM
    """
    from utils import calculate_psnr, calculate_ssim
    model.eval()

    valid_hr_list = sorted([
        os.path.join(valid_hr_path, f) for f in os.listdir(valid_hr_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    valid_lr_list = sorted([
        os.path.join(valid_lr_path, f) for f in os.listdir(valid_lr_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    total_psnr, total_ssim, count = 0, 0, 0

    from PIL import Image
    from torchvision import transforms

    logger.info("Evaluating on %d images..." % len(valid_hr_list))
    for hr_path, lr_path in zip(valid_hr_list, valid_lr_list):
        hr_img = transforms.ToTensor()(Image.open(hr_path).convert('RGB')).unsqueeze(0)
        lr_img = transforms.ToTensor()(Image.open(lr_path).convert('RGB')).unsqueeze(0)

        with torch.no_grad():
            sr_output = model(lr_img.to(device)).cpu().clamp_(0, 1)

        # 裁剪边界
        sr_np = sr_output.squeeze().numpy()
        hr_np = hr_img.squeeze().numpy()
        crop = cfg.TRAIN.scale_factor
        sr_np = sr_np[:, crop:-crop, crop:-crop]
        hr_np = hr_np[:, crop:-crop, crop:-crop]

        psnr = calculate_psnr(sr_np, hr_np)
        ssim = calculate_ssim(sr_np, hr_np)
        total_psnr += psnr
        total_ssim += ssim
        count += 1

    avg_psnr = total_psnr / max(count, 1)
    avg_ssim = total_ssim / max(count, 1)
    logger.info("Evaluation: PSNR: %.4f dB, SSIM: %.6f" % (avg_psnr, avg_ssim))
    return avg_psnr, avg_ssim


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SEFNet Training')
    parser.add_argument('--stage', type=str, default='all', choices=['1', '2', '3', 'all'])
    parser.add_argument('--hr_path', type=str, default=config.TRAIN.hr_img_path)
    parser.add_argument('--lr_path', type=str, default=None)
    parser.add_argument('--valid_hr_path', type=str, default=config.VALID.hr_img_path)
    parser.add_argument('--valid_lr_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=config.TRAIN.batch_size)
    parser.add_argument('--crop_size', type=int, default=config.TRAIN.crop_size)
    parser.add_argument('--scale', type=int, default=config.TRAIN.scale_factor)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s" % device)

    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    # 日志保存到文件
    fh = logging.FileHandler('log/sefnet_train_%s.log' % strftime("%Y%m%d_%H%M%S", localtime()))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    ###====================== PRE-LOAD DATA ===========================###
    dataset = SRDataset(
        hr_path=args.hr_path, lr_path=args.lr_path,
        crop_size=args.crop_size, scale_factor=args.scale,
        augment=True, repeat=10
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info("Dataset size: %d, DataLoader iters: %d" % (len(dataset), len(dataloader)))

    ###========================== DEFINE MODEL ============================###
    generator = arch.RRDB_Net(
        3, 3, config.TRAIN.nf, config.TRAIN.nb, gc=config.TRAIN.gc,
        upscale=args.scale, norm_type=None, act_type='leakyrelu', mode='CNA',
        upsample_mode='pixelshuffle',
        context_dim=config.TRAIN.context_dim,
        exposure_feat_dim=config.TRAIN.exposure_feat_dim
    ).to(device)
    logger.info("Generator params: {:,}".format(sum(p.numel() for p in generator.parameters())))

    discriminator = arch.Discriminator_VGG(in_nc=3, nf=64)
    logger.info("Discriminator params: {:,}".format(sum(p.numel() for p in discriminator.parameters())))

    # VGG感知损失
    vgg_net = arch.VGGFeatureExtractor(feature_layer=34).to(device)
    vgg_net.eval()

    # 固定样张输入（训练期间监控用）
    from PIL import Image
    from torchvision import transforms as T
    sample_dir = args.hr_path
    if os.path.isdir(sample_dir):
        sample_files = sorted([f for f in os.listdir(sample_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:batch_size]
        if sample_files:
            sample_imgs = torch.stack([
                T.ToTensor()(Image.open(os.path.join(sample_dir, f)).convert('RGB').resize(
                    (args.crop_size, args.crop_size), Image.BICUBIC))
                for f in sample_files
            ])
            sample_lr = F.interpolate(sample_imgs, scale_factor=1.0 / args.scale, mode='bicubic',
                                      align_corners=False).clamp_(0, 1)
        else:
            sample_lr = torch.rand(batch_size, 3, args.crop_size // args.scale,
                                   args.crop_size // args.scale)
    else:
        sample_lr = torch.rand(batch_size, 3, args.crop_size // args.scale,
                               args.crop_size // args.scale)

    # 保存样张参考
    save_image(sample_lr, 'samples/_train_sample_lr.png', nrow=ni)
    logger.info("Sample LR images saved to samples/_train_sample_lr.png")

    ###============================= TRAINING ===============================###
    if args.stage == '1' or args.stage == 'all':
        train_stage1(generator, dataloader, sample_lr, device, config)
    if args.stage == '2' or args.stage == 'all':
        train_stage2(generator, vgg_net, dataloader, sample_lr, device, config)
    if args.stage == '3' or args.stage == 'all':
        train_stage3(generator, discriminator, vgg_net, dataloader, sample_lr, device, config)

    # 保存最终模型
    final_path = os.path.join('checkpoint', 'SEFNet_G_final.pth')
    torch.save(generator.state_dict(), final_path)
    logger.info("Final model saved: %s" % final_path)

    # 评估
    if args.valid_hr_path and os.path.isdir(args.valid_hr_path):
        evaluate(generator, args.valid_hr_path, args.valid_lr_path, device, config)


if __name__ == '__main__':
    main()

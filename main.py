#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G (stage1)
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (stage2)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###========================== DEFINE MODEL ============================###
    net_g = RRDB_Net(3, 3, 64, config.TRAIN.nb, gc=32, upscale=4,
                     norm_type=None, act_type='leakyrelu', mode='CNA',
                     upsample_mode='pixelshuffle',
                     context_dim=config.TRAIN.context_dim,
                     exposure_feat_dim=config.TRAIN.exposure_feat_dim).to(device)
    net_d = Discriminator_VGG(in_nc=3, nf=64).to(device)
    vgg_net = VGGFeatureExtractor(feature_layer=34).to(device)
    vgg_net.eval()

    tl.layers.print_params(net_g, details=False)
    tl.layers.print_params(net_d, details=False)

    ## define losses
    criterion = MultiScaleExposureConsistencyLoss(
        vgg_net,
        lambda_pixel=config.TRAIN.lambda_pixel,
        lambda_percep=config.TRAIN.lambda_percep,
        lambda_consistency=config.TRAIN.lambda_consistency,
        lambda_exposure=config.TRAIN.lambda_exposure
    ).to(device)

    g_vars = tl.layers.get_variables_with_name(net_g, '', train_only=True)
    d_vars = tl.layers.get_variables_with_name(net_d, '', train_only=True)

    ## Pretrain optimizer
    g_optim_init = torch.optim.Adam(g_vars, lr=lr_init, betas=(beta1, 0.999))
    ## GAN optimizers
    g_optim = torch.optim.Adam(g_vars, lr=lr_init, betas=(beta1, 0.999))
    d_optim = torch.optim.Adam(d_vars, lr=lr_init, betas=(beta1, 0.999))

    ###========================== RESTORE MODEL =============================###
    if tl.files.load_and_assign_npz(name=checkpoint_dir + '/g_{}.pth'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(name=checkpoint_dir + '/g_{}_init.pth'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(name=checkpoint_dir + '/d_{}.pth'.format(tl.global_flag['mode']), network=net_d)

    ###========================= initialize G ====================###
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        for idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            step_time = time.time()
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            ## update G
            sr_output = net_g(lr_imgs)
            errM = F.l1_loss(sr_output, hr_imgs)

            g_optim_init.zero_grad()
            errM.backward()
            g_optim_init.step()

            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM.item()))
            total_mse_loss += errM.item()
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            with torch.no_grad():
                out = net_g(sample_lr.to(device))
            out = out.cpu().clamp_(0, 1)
            tl.vis.save_images(out.numpy(), [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)
            print("[*] save images")

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.state_dict(), name=checkpoint_dir + '/g_{}_init.pth'.format(tl.global_flag['mode']))

    ###========================= train GAN =========================###
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            for pg in g_optim.param_groups:
                pg['lr'] = lr_init * new_lr_decay
            for pg in d_optim.param_groups:
                pg['lr'] = lr_init * new_lr_decay
            log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            for pg in g_optim.param_groups:
                pg['lr'] = lr_init
            for pg in d_optim.param_groups:
                pg['lr'] = lr_init
            log = " ** init lr: %f  decay_every: %d, lr_decay: %f" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        for idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            step_time = time.time()
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            ## update D
            sr_output = net_g(lr_imgs).detach()
            logits_real = net_d(hr_imgs)
            logits_fake = net_d(sr_output)
            errD = relativistic_discriminator_loss(logits_real, logits_fake)

            d_optim.zero_grad()
            errD.backward()
            d_optim.step()

            ## update G
            sr_output = net_g(lr_imgs)
            logits_real = net_d(hr_imgs)
            logits_fake = net_d(sr_output)

            sr_lum = compute_luminance(sr_output)
            hr_lum = compute_luminance(hr_imgs)
            content_loss, loss_dict = criterion(sr_output, hr_imgs, sr_lum, hr_lum)
            gan_loss = relativistic_generator_loss(logits_real, logits_fake)
            errG = content_loss + config.TRAIN.lambda_gan * gan_loss

            g_optim.zero_grad()
            errG.backward()
            g_optim.step()

            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD.item(), errG.item(),
                   loss_dict['pixel'], loss_dict['percep'], gan_loss.item()))
            total_d_loss += errD.item()
            total_g_loss += errG.item()
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % \
              (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            with torch.no_grad():
                out = net_g(sample_lr.to(device))
            out = out.cpu().clamp_(0, 1)
            tl.vis.save_images(out.numpy(), [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
            print("[*] save images")

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.state_dict(), name=checkpoint_dir + '/g_{}.pth'.format(tl.global_flag['mode']))
            tl.files.save_npz(net_d.state_dict(), name=checkpoint_dir + '/d_{}.pth'.format(tl.global_flag['mode']))


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###========================== DEFINE MODEL ============================###
    net_g = RRDB_Net(3, 3, 64, config.TEST.nb, gc=32, upscale=4,
                     norm_type=None, act_type='leakyrelu', mode='CNA',
                     upsample_mode='pixelshuffle',
                     context_dim=config.TRAIN.context_dim,
                     exposure_feat_dim=config.TRAIN.exposure_feat_dim).to(device)

    ###========================== RESTORE G =============================###
    tl.files.load_and_assign_npz(name=checkpoint_dir + '/g_{}.pth'.format(tl.global_flag['mode']), network=net_g)

    ###======================= EVALUATION =============================###
    valid_lr_img = tl.vis.read_image('valid_lr.png', 'data2017/')
    valid_lr_img = valid_lr_img / 127.5 - 1  # rescale to [-1, 1]
    valid_lr_img = torch.from_numpy(valid_lr_img).float().permute(2, 0, 1).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        out = net_g(valid_lr_img)
    print("took: %4.4fs" % (time.time() - start_time))
    out = out.cpu().clamp_(0, 1).numpy()

    print("[*] save images")
    tl.vis.save_image(out[0].transpose(1, 2, 0), save_dir + '/valid_gen.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sefnet', help='sefnet, evaluate')
    parser.add_argument('--hr_path', type=str, default=config.TRAIN.hr_img_path)
    parser.add_argument('--lr_path', type=str, default=config.TRAIN.lr_img_path)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    ###====================== PRE-LOAD DATA ===========================###
    dataset = SRDataset(hr_path=args.hr_path, lr_path=args.lr_path,
                        crop_size=config.TRAIN.crop_size,
                        scale_factor=config.TRAIN.scale_factor,
                        augment=True, repeat=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    ## sample images for monitoring
    from PIL import Image
    from torchvision import transforms as T
    sample_files = sorted([f for f in os.listdir(args.hr_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:batch_size]
    sample_imgs = torch.stack([
        T.ToTensor()(Image.open(os.path.join(args.hr_path, f)).convert('RGB').resize(
            (config.TRAIN.crop_size, config.TRAIN.crop_size), Image.BICUBIC))
        for f in sample_files
    ])
    sample_lr = F.interpolate(sample_imgs, scale_factor=1.0 / config.TRAIN.scale_factor,
                              mode='bicubic', align_corners=False).clamp_(0, 1)

    if tl.global_flag['mode'] == 'sefnet':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")

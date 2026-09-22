import sys
import torch
from utils import load_checkpoint, ARCH_DEFAULTS

alpha = float(sys.argv[1])

net_PSNR_path = './checkpoint/g_sefnet_stage2.pth'
net_GAN_path = './checkpoint/g_sefnet_stage3.pth'
net_interp_path = './checkpoint/interp_{:02d}.pth'.format(int(alpha * 10))

state_psnr, meta_psnr = load_checkpoint(net_PSNR_path, map_location='cpu')
state_gan, _ = load_checkpoint(net_GAN_path, map_location='cpu')

net_interp = {}
print('Interpolating with alpha =', alpha)

for k, v_psnr in state_psnr.items():
    v_gan = state_gan[k]
    net_interp[k] = (1 - alpha) * v_psnr + alpha * v_gan

torch.save({
    'state_dict': net_interp,
    'arch': meta_psnr.get('arch', dict(ARCH_DEFAULTS)),
    'alpha': alpha,
}, net_interp_path)
print('Saved:', net_interp_path)

import sys
import torch
from collections import OrderedDict

alpha = float(sys.argv[1])

net_PSNR_path = './checkpoint/g_sefnet_stage2.pth'
net_GAN_path = './checkpoint/g_sefnet_stage3.pth'
net_interp_path = './checkpoint/interp_{:02d}.pth'.format(int(alpha*10))

net_PSNR = torch.load(net_PSNR_path)
net_GAN = torch.load(net_GAN_path)
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)

for k, v_PSNR in net_PSNR.items():
    v_GAN = net_GAN[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_GAN

torch.save(net_interp, net_interp_path)

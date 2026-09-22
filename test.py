import sys
import os
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
from utils import load_checkpoint, build_generator, ARCH_DEFAULTS

model_path = sys.argv[1]  # checkpoint/SEFNet_G_final.pth
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_img_folder = sys.argv[2] if len(sys.argv) > 2 else 'LR/*'
out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

state_dict, meta = load_checkpoint(model_path, map_location=device)
arch_cfg = meta.get('arch', dict(ARCH_DEFAULTS))
# 兼容旧 checkpoint：强制与训练默认 nb=16 对齐
arch_cfg.setdefault('nb', 16)
arch_cfg.setdefault('nf', 64)
arch_cfg.setdefault('gc', 32)
arch_cfg.setdefault('upscale', 4)
arch_cfg.setdefault('context_dim', 64)
arch_cfg.setdefault('exposure_feat_dim', 32)

model = build_generator(arch_cfg)
model.load_state_dict(state_dict, strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path %s (nb=%d, nf=%d).' % (model_path, arch_cfg['nb'], arch_cfg['nf']))
print('Testing...')

idx = 0
paths = sorted(glob.glob(test_img_folder))
if not paths:
    print('No images matched %r' % test_img_folder)
    sys.exit(1)

for path in paths:
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print('  skip (unreadable)')
        continue
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, '%s_rlt.png' % base), output)

print('Done. %d image(s) -> %s/' % (idx, out_dir))

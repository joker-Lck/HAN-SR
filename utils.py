import torch
import numpy as np
from PIL import Image

# 统一网络结构默认值（训练/推理必须一致，否则 state_dict 对不上）
ARCH_DEFAULTS = {
    'in_nc': 3,
    'out_nc': 3,
    'nf': 64,
    'nb': 16,
    'gc': 32,
    'upscale': 4,
    'context_dim': 64,
    'exposure_feat_dim': 32,
    'upsample_mode': 'pixelshuffle',
    'norm_type': None,
    'act_type': 'leakyrelu',
    'mode': 'CNA',
    'grad_checkpoint': False,
}


def build_generator(arch_cfg=None):
    from architecture import RRDB_Net
    cfg = {**ARCH_DEFAULTS, **(arch_cfg or {})}
    return RRDB_Net(
        cfg['in_nc'], cfg['out_nc'], cfg['nf'], cfg['nb'], gc=cfg['gc'],
        upscale=cfg['upscale'], norm_type=cfg['norm_type'],
        act_type=cfg['act_type'], mode=cfg['mode'],
        upsample_mode=cfg['upsample_mode'],
        context_dim=cfg['context_dim'],
        exposure_feat_dim=cfg['exposure_feat_dim'],
        grad_checkpoint=cfg.get('grad_checkpoint', False),
    )


def save_checkpoint(model, path, optimizer=None, epoch=0, loss=None, arch_cfg=None):
    """统一 checkpoint 格式: {'state_dict', 'arch', 'epoch', 'loss', 'optimizer'?}"""
    payload = {
        'state_dict': model.state_dict(),
        'arch': {**ARCH_DEFAULTS, **(arch_cfg or {})},
        'epoch': epoch,
        'loss': loss,
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, map_location='cpu'):
    """
    兼容三种历史格式，返回 (state_dict, meta):
      1. 新格式: {'state_dict': ..., 'arch': ...}
      2. train.py 旧格式: {'model_state_dict': ...}
      3. 裸 state_dict
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict'], ckpt
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        meta = dict(ckpt)
        meta.setdefault('arch', ARCH_DEFAULTS)
        return ckpt['model_state_dict'], meta
    if isinstance(ckpt, dict) and len(ckpt) > 0 and all(hasattr(v, 'shape') for v in ckpt.values()):
        return ckpt, {'state_dict': ckpt, 'arch': dict(ARCH_DEFAULTS)}
    raise ValueError('Unrecognized checkpoint format: %r' % (type(ckpt),))


class AmpHelper:
    """
    混合精度助手（面向 16GB 卡）。
    mode: 'auto' | 'on' | 'off'
      auto: CUDA 优先 bf16，否则 fp16；CPU 关闭
    fp16 使用 GradScaler；bf16 不用 scaler（动态范围同 fp32）。
    """

    def __init__(self, device, mode='auto'):
        self.device = device
        use_cuda = device.type == 'cuda'
        if mode == 'off' or not use_cuda:
            self.enabled = False
            self.dtype = torch.float32
            self.scaler = None
            self.mode_name = 'off'
            return
        self.enabled = True
        if mode == 'on' or mode == 'auto':
            bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            self.dtype = torch.float32
            self.enabled = False
        if self.enabled and self.dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        self.mode_name = 'bf16' if self.dtype == torch.bfloat16 else 'fp16'

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.enabled)

    def backward(self, loss, accum_scale=1.0):
        loss = loss / accum_scale
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer, params=None, max_norm=1.0):
        if self.scaler is not None:
            if params is not None and max_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if params is not None and max_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def get_imgs_fn(file_name, path):
    """Input an image path and name, return an image array"""
    return np.array(Image.open(path + file_name).convert('RGB'))


def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def crop(x, wrg=384, hrg=384, is_random=True):
    h, w = x.shape[0], x.shape[1]
    if is_random:
        top = np.random.randint(0, max(h - hrg, 1))
        left = np.random.randint(0, max(w - wrg, 1))
    else:
        top = max((h - hrg) // 2, 0)
        left = max((w - wrg) // 2, 0)
    return x[top:top + hrg, left:left + wrg].copy()


def imresize(x, size=[96, 96], interp='bicubic', mode=None):
    img = Image.fromarray(x.astype(np.uint8))
    method = Image.BICUBIC if interp == 'bicubic' else Image.BILINEAR
    img = img.resize((size[1], size[0]), method)
    return np.array(img).astype(x.dtype)


def calculate_psnr(sr, hr):
    sr = np.clip(sr, 0, 1)
    hr = np.clip(hr, 0, 1)
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def calculate_ssim(sr, hr, win_size=7):
    from scipy.ndimage import gaussian_filter
    sr = np.clip(sr, 0, 1).astype(np.float64)
    hr = np.clip(hr, 0, 1).astype(np.float64)
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    ssims = []
    for ch in range(sr.shape[0]):
        mu1 = gaussian_filter(sr[ch], sigma=1.5, truncate=(win_size-1)/2.0/1.5)
        mu2 = gaussian_filter(hr[ch], sigma=1.5, truncate=(win_size-1)/2.0/1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(sr[ch] ** 2, sigma=1.5, truncate=(win_size-1)/2.0/1.5) - mu1_sq
        sigma2_sq = gaussian_filter(hr[ch] ** 2, sigma=1.5, truncate=(win_size-1)/2.0/1.5) - mu2_sq
        sigma12 = gaussian_filter(sr[ch] * hr[ch], sigma=1.5, truncate=(win_size-1)/2.0/1.5) - mu1_mu2
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssims.append((numerator / denominator).mean())
    return np.mean(ssims)

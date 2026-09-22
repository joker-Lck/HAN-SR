from easydict import EasyDict as edict
import json

config = edict()

config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9
config.TRAIN.beta2 = 0.999

## initialize G
config.TRAIN.n_epoch_init = 100

## three-stage schedule (aligned with train_sefnet.py defaults)
config.TRAIN.stage1_lr = 1e-4
config.TRAIN.stage2_lr = 1e-4
config.TRAIN.stage3_lr = 1e-4
config.TRAIN.stage1_epochs = 50
config.TRAIN.stage2_epochs = 200
config.TRAIN.stage3_epochs = 100

## adversarial learning (SEFNet)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 50

## loss weights
config.TRAIN.lambda_pixel = 1.0
config.TRAIN.lambda_percep = 1.0
config.TRAIN.lambda_consistency = 0.5
config.TRAIN.lambda_exposure = 0.3
config.TRAIN.lambda_gan = 1e-3
config.TRAIN.lambda_calibration = 0.2  # 曝光校准损失（普通曝光+渐变曝光）

## network params
config.TRAIN.nf = 64
config.TRAIN.nb = 16
config.TRAIN.gc = 32
config.TRAIN.context_dim = 64
config.TRAIN.exposure_feat_dim = 32

## train set location
config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.TRAIN.crop_size = 128
config.TRAIN.scale_factor = 4

config.TRAIN.log_interval = 100

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

config.TEST = edict()
# 与训练保持一致（nb 不同会导致 state_dict 无法 strict 加载）
config.TEST.nb = 16

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")

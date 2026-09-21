import math
import torch
import torch.nn as nn
import block as B


class RRDB_Net(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
            mode='CNA', res_scale=1, upsample_mode='upconv', \
            context_dim=64, exposure_feat_dim=32):
        super(RRDB_Net, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        # exposure analysis modules
        self.exposure_stats = B.GlobalExposureStatistics(in_nc, context_dim)
        self.relative_encoder = B.RelativeExposureEncoder(in_nc, context_dim, exposure_feat_dim)

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.AdaptiveRRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA', exposure_feat_dim=exposure_feat_dim) \
            for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.conv_first = fea_conv
        self.rrdb_blocks = nn.ModuleList(rb_blocks)
        self.LR_conv = LR_conv
        self.upsample = B.sequential(*upsampler)
        self.HR_conv0 = HR_conv0
        self.HR_conv1 = HR_conv1

    def forward(self, x):
        context, global_feat = self.exposure_stats(x)
        exposure_feat = self.relative_encoder(x, context)

        feat = self.conv_first(x)
        body_feat = feat
        for block in self.rrdb_blocks:
            body_feat = block(body_feat, exposure_feat)
        body_feat = self.LR_conv(body_feat)
        feat = feat + body_feat

        feat = self.upsample(feat)
        feat = self.HR_conv0(feat)
        out = self.HR_conv1(feat)
        return out


class Discriminator_VGG(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf*2, 3, 1, 1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2, nf*2, 3, 2, 1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2, nf*4, 3, 1, 1),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4, nf*4, 3, 2, 1),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4, nf*8, 3, 1, 1),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*8, nf*8, 3, 2, 1),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf*8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34):
        super(VGGFeatureExtractor, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        try:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except TypeError:
            vgg = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

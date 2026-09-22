from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


####################
# Exposure-Aware blocks
####################


class GlobalExposureStatistics(nn.Module):
    """
    全局曝光统计模块
    从输入图像中提取全局曝光统计信息
    输出一个紧凑的上下文向量，描述整张图的曝光分布特征

    结构: 3层下采样卷积 -> 全局均值+标准差池化 -> FC映射
    不仅统计均值，还统计标准差，捕获亮度分布的离散程度
    """
    def __init__(self, in_channels=3, context_dim=64):
        super(GlobalExposureStatistics, self).__init__()
        # 轻量编码器提取多尺度统计
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 全局统计池化（同时用均值和标准差）
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 统计量 -> 上下文向量
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),  # *2 因为均值+标准差
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, context_dim),
        )

    def forward(self, x):
        """
        x: [B, 3, H, W]
        返回: context [B, context_dim], feat [B, 64, h, w]
        """
        feat = self.encoder(x)  # [B, 64, h, w]
        mean_feat = self.pool(feat)  # [B, 64, 1, 1]
        std_feat = torch.sqrt(self.pool((feat - mean_feat) ** 2) + 1e-6)  # [B, 64, 1, 1]
        stats = torch.cat([mean_feat.flatten(1), std_feat.flatten(1)], dim=1)  # [B, 128]
        context = self.fc(stats)  # [B, context_dim]
        return context, feat


class RelativeExposureEncoder(nn.Module):
    """
    相对曝光编码器

    核心思想:
    - 绝对亮度在不同图片间差异大，不好泛化
    - 相对位置（这个区域比全局平均亮多少/暗多少）更有意义
    - 还要编码"渐变方向"信息（上->下变暗？还是中心->边缘变暗？）

    结构: 局部特征提取 + 正弦空间位置编码 + 全局上下文广播 -> 融合卷积
    """
    def __init__(self, in_channels=3, context_dim=64, feat_dim=32):
        super(RelativeExposureEncoder, self).__init__()
        # 局部亮度统计提取器
        self.local_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, feat_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 空间位置编码（让网络知道每个像素在图中的位置）
        # 对识别渐变方向至关重要
        self.pos_encoding = nn.Parameter(
            self._create_sinusoidal_encoding(256, 256, feat_dim // 2),
            requires_grad=False
        )
        # 相对曝光计算：融合局部特征 + 位置信息 + 全局上下文
        self.relative_head = nn.Sequential(
            nn.Conv2d(feat_dim + feat_dim // 2 + context_dim, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, feat_dim, 1),
        )

    def _create_sinusoidal_encoding(self, h, w, dim):
        """
        正弦位置编码：类似Transformer的位置编码
        让网络知道每个像素的空间位置，从而感知渐变方向

        通道布局: [sin_h_0, cos_h_0, sin_h_1, cos_h_1, ...,
                   sin_w_0, cos_w_0, sin_w_1, cos_w_1, ...]
        前半部分编码高度位置，后半部分编码宽度位置，各占 dim/2 通道。

        修复说明:
        原代码 pe[half_dim::2] 起始索引超过通道数，宽度编码完全丢失。
        同时修正了维度广播顺序（pos[h, freq] -> pe[freq, h, w]）。
        """
        pe = torch.zeros(dim, h, w)
        half_dim = dim // 2  # 高度和宽度各占一半通道
        num_freq = half_dim // 2  # sin/cos成对，各 num_freq 个频率
        div_term = torch.exp(
            torch.arange(0, num_freq).float() * (-math.log(10000.0) / num_freq)
        )
        # 高度编码：pe[0:half_dim]，共 half_dim 通道（num_freq个sin + num_freq个cos）
        pos_h = torch.arange(0, h).float().unsqueeze(1)  # [h, 1]
        sin_h = torch.sin(pos_h * div_term)  # [h, num_freq]
        cos_h = torch.cos(pos_h * div_term)  # [h, num_freq]
        # 交错排列: pe[0]=sin_h_0, pe[1]=cos_h_0, pe[2]=sin_h_1, pe[3]=cos_h_1, ...
        pe[0:half_dim:2, :, :] = sin_h.permute(1, 0).unsqueeze(2).expand(-1, -1, w)
        pe[1:half_dim:2, :, :] = cos_h.permute(1, 0).unsqueeze(2).expand(-1, -1, w)

        # 宽度编码：pe[half_dim:dim]，共 half_dim 通道
        pos_w = torch.arange(0, w).float().unsqueeze(1)  # [w, 1]
        sin_w = torch.sin(pos_w * div_term)  # [w, num_freq]
        cos_w = torch.cos(pos_w * div_term)  # [w, num_freq]
        pe[half_dim:dim:2, :, :] = sin_w.permute(1, 0).unsqueeze(1).expand(-1, h, -1)
        pe[half_dim + 1:dim:2, :, :] = cos_w.permute(1, 0).unsqueeze(1).expand(-1, h, -1)
        return pe.unsqueeze(0)  # [1, dim, h, w]

    def forward(self, x, context):
        """
        x: [B, 3, H, W] 输入图像
        context: [B, context_dim] 全局曝光上下文
        返回: [B, feat_dim, H, W] 相对曝光特征图
        """
        B, _, H, W = x.shape
        local_feat = self.local_encoder(x)  # [B, feat_dim, h', w']
        # 修复：位置编码插值到local_feat的实际尺寸（而非原始输入尺寸），
        # 避免H/W不是8倍数时尺寸不匹配
        _, _, lh, lw = local_feat.shape
        pos = F.interpolate(
            self.pos_encoding.expand(B, -1, -1, -1),
            size=(lh, lw), mode='bilinear', align_corners=False
        )  # [B, feat_dim//2, lh, lw]
        # 全局上下文广播到local_feat的空间尺寸
        context_map = context.unsqueeze(2).unsqueeze(3).expand(-1, -1, lh, lw)
        # 融合：局部特征 + 位置 + 全局上下文
        combined = torch.cat([local_feat, pos, context_map], dim=1)
        relative_feat = self.relative_head(combined)  # [B, feat_dim, lh, lw]
        # 插值回原始输入尺寸，保证与主干网络特征图对齐
        if lh != H or lw != W:
            relative_feat = F.interpolate(
                relative_feat, size=(H, W), mode='bilinear', align_corners=False
            )
        return relative_feat


class ExposureAdaptiveDeformConv(nn.Module):
    """
    亮度自适应可变形卷积

    标准可变形卷积：偏移量只依赖空间特征
    本方法：偏移量同时依赖空间特征和局部曝光信息

    效果:
    - 暗区：卷积核自动扩大感受野，收集更多上下文信息来降噪
    - 亮区：卷积核自动缩小感受野，聚焦局部细节来保真
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 exposure_feat_dim=32, padding=1):
        super(ExposureAdaptiveDeformConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = 1
        self.dilation = 1

        offset_in = in_channels + exposure_feat_dim
        # 偏移量预测：同时看特征和曝光信息
        # 输出 2 * kernel_size^2 个值（每个采样点的x,y偏移）
        self.offset_conv = nn.Sequential(
            nn.Conv2d(offset_in, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 2 * kernel_size * kernel_size, 3, padding=1),
        )
        # 调制掩码预测：控制每个采样点的权重
        # 输出 kernel_size^2 个值（0~1之间的权重）
        self.mask_conv = nn.Sequential(
            nn.Conv2d(offset_in, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, kernel_size * kernel_size, 3, padding=1),
            nn.Sigmoid(),
        )
        # 实际的卷积权重
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)
        # 修复：偏移量预测最后一层必须零初始化，使初始偏移为0（可变形卷积退化为标准卷积）
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)
        # 调制掩码最后一层初始化为0.5（初始等权融合），确保训练起点稳定
        # 注意：mask_conv最后一层是Sigmoid，所以这里初始化前一层使其输出接近0.5
        nn.init.zeros_(self.mask_conv[-2].weight)
        nn.init.zeros_(self.mask_conv[-2].bias)

    def forward(self, x, exposure_feat):
        """
        x: [B, C_in, H, W] 输入特征
        exposure_feat: [B, C_exp, H, W] 曝光特征
        返回: [B, C_out, H, W]
        """
        from torchvision.ops import deform_conv2d
        # 拼接特征和曝光信息
        combined = torch.cat([x, exposure_feat], dim=1)
        # 预测偏移量和调制掩码
        offset = self.offset_conv(combined)   # [B, 2*K*K, H, W]
        mask = self.mask_conv(combined)        # [B, K*K, H, W]
        # 可变形卷积
        out = deform_conv2d(
            input=x, offset=offset, weight=self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask
        )
        return out


class SEFConvBlock(nn.Module):
    """
    曝光自适应卷积块

    双路径设计:
    - 标准路径：处理亮度无关的通用特征
    - 自适应路径：专门处理曝光相关的特征变化
    - 门控融合：根据曝光情况自动调节两条路径的贡献比例
    """
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='leakyrelu',
                 mode='CNA', exposure_feat_dim=32):
        super(SEFConvBlock, self).__init__()
        padding = get_valid_padding(kernel_size, 1)
        p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
        padding = padding if pad_type == 'zero' else 0

        # 标准卷积路径
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.act = act(act_type) if act_type else None
        self.norm = norm(norm_type, out_nc) if norm_type else None

        # 曝光自适应可变形路径
        self.deform_conv = ExposureAdaptiveDeformConv(
            in_nc, out_nc, kernel_size=kernel_size,
            exposure_feat_dim=exposure_feat_dim, padding=padding
        )
        # 门控融合：自动决定两条路径的权重
        # 修复：使用3x3卷积保留空间信息，而非全局池化（原代码池化后expand导致门控空间均匀）
        self.gate = nn.Sequential(
            nn.Conv2d(in_nc + exposure_feat_dim, max(in_nc // 4, 8), 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(in_nc // 4, 8), out_nc, 3, padding=1),
            nn.Sigmoid()
        )
        self.pad = p

    def forward(self, x, exposure_feat):
        """
        x: [B, C_in, H, W]
        exposure_feat: [B, C_exp, H, W]
        """
        # 标准路径
        std_out = x
        if self.pad is not None:
            std_out = self.pad(std_out)
        std_out = self.conv(std_out)
        if self.norm is not None:
            std_out = self.norm(std_out)
        if self.act is not None:
            std_out = self.act(std_out)

        # 自适应路径
        adapt_out = self.deform_conv(x, exposure_feat)
        if self.act is not None:
            adapt_out = self.act(adapt_out)

        # 门控融合（修复：直接拼接x和exposure_feat，保留空间分辨率）
        gate_input = torch.cat([x, exposure_feat], dim=1)
        g = self.gate(gate_input)  # [B, C_out, H, W]
        # g趋近1时偏向标准路径，趋近0时偏向自适应路径
        return g * std_out + (1 - g) * adapt_out


class AdaptiveResidualDenseBlock_5C(nn.Module):
    """
    曝光自适应残差密集块（改进的RDB）
    style: 5 convs, 前4层用SEFConvBlock，第5层标准卷积
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', exposure_feat_dim=32):
        super(AdaptiveResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = SEFConvBlock(nc, gc, kernel_size, stride, bias, pad_type,
                                  norm_type, act_type, mode, exposure_feat_dim)
        self.conv2 = SEFConvBlock(nc + gc, gc, kernel_size, stride, bias, pad_type,
                                  norm_type, act_type, mode, exposure_feat_dim)
        self.conv3 = SEFConvBlock(nc + 2 * gc, gc, kernel_size, stride, bias, pad_type,
                                  norm_type, act_type, mode, exposure_feat_dim)
        self.conv4 = SEFConvBlock(nc + 3 * gc, gc, kernel_size, stride, bias, pad_type,
                                  norm_type, act_type, mode, exposure_feat_dim)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        # 最后一层用标准卷积（不需要曝光自适应）
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x, exposure_feat):
        x1 = self.conv1(x, exposure_feat)
        x2 = self.conv2(torch.cat((x, x1), 1), exposure_feat)
        x3 = self.conv3(torch.cat((x, x1, x2), 1), exposure_feat)
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1), exposure_feat)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class AdaptiveRRDB(nn.Module):
    """
    曝光自适应 RRDB（Residual in Residual Dense Block）
    3个级联的AdaptiveRDB，残差缩放
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', exposure_feat_dim=32):
        super(AdaptiveRRDB, self).__init__()
        self.RDB1 = AdaptiveResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode, exposure_feat_dim)
        self.RDB2 = AdaptiveResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode, exposure_feat_dim)
        self.RDB3 = AdaptiveResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode, exposure_feat_dim)

    def forward(self, x, exposure_feat):
        out = self.RDB1(x, exposure_feat)
        out = self.RDB2(out, exposure_feat)
        out = self.RDB3(out, exposure_feat)
        return out.mul(0.2) + x


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                        pad_type='zero', norm_type=None, act_type='relu'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)


####################
# 损失函数
####################


class MultiScaleExposureConsistencyLoss(nn.Module):
    """
    多尺度曝光一致性损失

    不只监督最终输出，还在VGG特征空间中监督曝光一致性：
    - pixel loss: L1像素损失
    - perceptual loss: VGG19特征损失（激活前特征）
    - exposure consistency loss: 曝光一致性损失
    - brightness distribution loss: 亮度分布损失
    """
    def __init__(self, vgg_net, lambda_pixel=1.0, lambda_percep=1.0,
                 lambda_consistency=0.5, lambda_exposure=0.3):
        super(MultiScaleExposureConsistencyLoss, self).__init__()
        self.vgg = vgg_net
        self.l1 = nn.L1Loss()
        self.lambda_pixel = lambda_pixel
        self.lambda_percep = lambda_percep
        self.lambda_consistency = lambda_consistency
        self.lambda_exposure = lambda_exposure

    def exposure_consistency_loss(self, sr_feat, hr_feat, sr_exposure, hr_exposure):
        """
        曝光一致性损失：同材质区域特征应一致，曝光差异大时降低约束权重

        修复：原代码用exp(-|diff|*5)，当diff很小时权重≈1，diff很大时权重≈0，
        导致中等曝光差异区域（最需要监督的区域）权重过低。
        改用sigmoid软阈值：diff<0.3时权重≈1，diff>0.7时权重≈0，中间平滑过渡。
        """
        target_size = sr_feat.shape[2:]
        sr_exp_resized = F.adaptive_avg_pool2d(sr_exposure, target_size)
        hr_exp_resized = F.adaptive_avg_pool2d(hr_exposure, target_size)
        exposure_diff = torch.abs(sr_exp_resized - hr_exp_resized)
        # sigmoid软阈值替代exp衰减
        weight = torch.sigmoid(5.0 - exposure_diff * 10)
        feat_diff = torch.abs(sr_feat - hr_feat)
        weighted_diff = feat_diff * weight
        return torch.mean(weighted_diff)

    def brightness_distribution_loss(self, sr_output, hr):
        """亮度分布损失：分块统计亮度的均值和标准差对比"""
        block_size = 16
        def get_block_stats(img):
            B, C, H, W = img.shape
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
            h_blocks = H // block_size
            w_blocks = W // block_size
            gray = gray[:, :, :h_blocks * block_size, :w_blocks * block_size]
            blocks = gray.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
            block_mean = blocks.mean(dim=(-1, -2))
            block_std = blocks.std(dim=(-1, -2))
            return block_mean, block_std

        sr_mean, sr_std = get_block_stats(sr_output)
        hr_mean, hr_std = get_block_stats(hr)
        loss_mean = self.l1(sr_mean, hr_mean)
        loss_std = self.l1(sr_std, hr_std)
        return loss_mean + loss_std

    def forward(self, sr_output, hr, sr_exposure, hr_exposure):
        """
        sr_output: [B, 3, H, W] 超分输出
        hr: [B, 3, H, W] 高分辨率GT
        sr_exposure: [B, 1, H, W] 超分输出的亮度
        hr_exposure: [B, 1, H, W] GT的亮度
        """
        # 1. 像素级L1损失
        loss_pixel = self.l1(sr_output, hr)
        if self.vgg is not None:
            # 2. 感知损失：GT 无梯度，避免为 HR 存 VGG 激活
            sr_vgg = self.vgg(sr_output)
            with torch.no_grad():
                hr_vgg = self.vgg(hr)
            loss_percep = self.l1(sr_vgg, hr_vgg)
            # 3. 曝光一致性损失（在VGG特征空间中）
            loss_consistency = self.exposure_consistency_loss(
                sr_vgg, hr_vgg, sr_exposure, hr_exposure
            )
        else:
            loss_percep = torch.zeros((), device=sr_output.device)
            loss_consistency = torch.zeros((), device=sr_output.device)
        # 4. 亮度分布损失
        loss_exposure = self.brightness_distribution_loss(sr_output, hr)

        total = (self.lambda_pixel * loss_pixel +
                 self.lambda_percep * loss_percep +
                 self.lambda_consistency * loss_consistency +
                 self.lambda_exposure * loss_exposure)
        return total, {
            'pixel': loss_pixel.item(),
            'percep': loss_percep.item(),
            'consistency': loss_consistency.item(),
            'exposure': loss_exposure.item()
        }


def compute_luminance(img):
    """计算图像亮度通道 (ITU-R BT.601)"""
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def relativistic_discriminator_loss(logits_real, logits_fake):
    """相对判别器损失"""
    loss_real = F.softplus(-(logits_real - logits_fake.mean())).mean()
    loss_fake = F.softplus(logits_fake - logits_real.mean()).mean()
    return (loss_real + loss_fake) / 2


def relativistic_generator_loss(logits_real, logits_fake):
    """相对生成器损失"""
    loss_real = F.softplus(-(logits_real - logits_fake.mean())).mean()
    loss_fake = F.softplus(logits_fake - logits_real.mean()).mean()
    return (loss_real + loss_fake) / 2

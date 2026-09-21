# SEFNet: Self-calibrating Exposure-aware Fusion Network

自曝光校准超分网络 — 针对曝光渐变图像的超分辨率算法

## 核心特性

| # | 特性 | 说明 |
|---|--------|------|
| 1 | **相对曝光编码** | 不学绝对亮度，学"相对于全局的曝光位置"，泛化性更好 |
| 2 | **亮度自适应可变形卷积** | 卷积核根据局部亮度自动调整采样策略（暗区扩大感受野、亮区聚焦细节） |
| 3 | **多尺度曝光一致性损失** | 在多个特征尺度监督曝光归一化效果 |
| 4 | **三阶段训练策略** | 曝光模块预训练 → 联合训练 → GAN对抗训练 |

## 项目结构

```
HAN-SR/
├── architecture.py   # 网络架构: RRDB_Net, Discriminator_VGG, VGGFeatureExtractor
├── block.py          # 基础模块 + 曝光自适应模块 + 损失函数 (核心代码)
├── config.py         # 超参数配置
├── data.py           # 数据集加载、数据增强、曝光渐变合成器
├── utils.py          # 工具函数: PSNR/SSIM计算、图像处理
├── train.py          # 三阶段训练脚本 (train + evaluate)
├── test.py           # 推理脚本
├── net_interp.py     # 模型插值工具
├── requirements.txt  # 依赖
└── README.md
```

## 环境依赖

```bash
pip install -r requirements.txt
```

## 数据准备

```
data/
├── DIV2K_train_HR/            # 训练集高分辨率图片
├── DIV2K_train_LR_bicubic/X4/ # 训练集低分辨率图片（可选）
├── DIV2K_valid_HR/            # 验证集高分辨率图片
└── DIV2K_valid_LR_bicubic/X4/ # 验证集低分辨率图片（可选）
```

## 训练

```bash
# 全流程三阶段训练
python train.py --stage all --hr_path data/DIV2K_train_HR/

# 分阶段训练
python train.py --stage 1 --hr_path data/DIV2K_train_HR/
python train.py --stage 2 --hr_path data/DIV2K_train_HR/
python train.py --stage 3 --hr_path data/DIV2K_train_HR/
```

训练过程中会自动:
- 保存样张到 `samples/`
- 保存checkpoint到 `checkpoint/`
- 写入训练日志到 `log/`
- 在验证集上评估PSNR/SSIM

## 推理

```bash
# 单张图片
python test.py checkpoint/SEFNet_G_final.pth
```

## 模型插值

```bash
# 在PSNR模型和GAN模型之间插值，alpha=0~1
python net_interp.py 0.8
```

## 网络架构

```
输入LR图（曝光渐变）
    │
    ├──→ 全局曝光统计模块 ──→ 曝光上下文向量 (64-d)
    │                              │
    ├──→ 相对曝光编码器 ──→ 逐像素曝光特征图 (32-ch)
    │         (局部特征 + 正弦位置编码 + 全局上下文)
    │
    └──→ 浅层特征提取 (Conv 64)
              │
              ├── AdaptiveRRDB × N (16 or 23 blocks)
              │     ├── 标准密集卷积路径
              │     ├── 亮度自适应可变形卷积路径
              │     └── 门控融合 (自动调节两条路径权重)
              │
              ├── 全局残差连接
              ├── PixelShuffle 上采样 (×4)
              └── 重建层 → 输出超分结果
```

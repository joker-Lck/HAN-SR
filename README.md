# SEFNet: Self-calibrating Exposure-aware Fusion Network

自曝光校准超分网络 — 针对曝光渐变图像的超分辨率算法

## 核心特性

| # | 特性 | 说明 |
|---|--------|------|
| 1 | **相对曝光编码** | 不学绝对亮度，学"相对于全局的曝光位置"，泛化性更好 |
| 2 | **亮度自适应可变形卷积** | 卷积核根据局部亮度自动调整采样策略（暗区扩大感受野、亮区聚焦细节） |
| 3 | **多尺度曝光一致性损失** | 在多个特征尺度监督曝光归一化效果 |
| 4 | **三阶段训练策略** | 内容预热 → 曝光一致性联合训练 → GAN对抗训练 |

## 项目结构

```
HAN-SR/
├── architecture.py   # 网络架构: RRDB_Net, Discriminator_VGG, VGGFeatureExtractor
├── block.py          # 基础模块 + 曝光自适应模块 + 损失函数 (核心代码)
├── config.py         # 超参数配置
├── data.py           # 数据集加载、数据增强、曝光渐变合成器
├── utils.py          # 工具: PSNR/SSIM、checkpoint 读写、build_generator
├── train_sefnet.py   # 【主入口】三阶段训练脚本
├── test.py           # 推理脚本
├── net_interp.py     # 模型插值工具
├── requirements.txt  # 依赖
└── README.md
```

> 训练请统一使用 `train_sefnet.py`。`train.py` 为旧版脚本，配置不完整。

## 环境依赖

```bash
pip install -r requirements.txt
```

Stage 2/3 使用 VGG19 感知损失。权重获取顺序：
1. 本地 `weights/vgg19-dcbb9e9d.pth`
2. 自动下载 ImageNet 预训练
3. 失败时退回随机初始化（仅警告，不阻塞；感知损失会变弱）

Stage 1 可完全离线运行。

## 硬件建议（RTX 5080 16GB）

| 配置 | 建议值 |
|------|--------|
| `batch_size` | 8–16 |
| `patch_size` | 128（可试 192） |
| `nb` | 16（训练默认） |
| 预估显存 | batch=8, patch=128, nb=16 约 6–8 GB；batch=16 约 12–14 GB |

## 数据准备

任意图片目录即可（在线合成曝光退化），推荐 DIV2K 训练集 HR 图：

```
data/
└── DIV2K_train_HR/            # 训练集高分辨率图片
```

## 训练

```bash
# 演示：小数据集快速跑通
python train_sefnet.py --hr_dir ./train_images --stage1_epochs 5 --stage2_epochs 5 --stage3_epochs 5

# 正式训练（DIV2K）
python train_sefnet.py --hr_dir data/DIV2K_train_HR --stage1_epochs 50 --stage2_epochs 200 --stage3_epochs 100

# 分阶段
python train_sefnet.py --hr_dir data/DIV2K_train_HR --stage 1
python train_sefnet.py --hr_dir data/DIV2K_train_HR --stage 2
python train_sefnet.py --hr_dir data/DIV2K_train_HR --stage 3
```

训练过程中会自动:
- 在线合成曝光退化（渐变/均匀/混合）
- 保存样张到 `samples/`
- 保存checkpoint到 `checkpoint/`（统一格式，含 arch 元信息）
- 写入训练日志到 `log/`

## 推理

```bash
# 默认读 LR/ 目录，结果写 results/
python test.py checkpoint/SEFNet_G_final.pth

# 指定输入图目录/glob
python test.py checkpoint/SEFNet_G_final.pth "data/DIV2K_valid_LR_bicubic/X4/*"
```

网络规模与训练保持一致（默认 `nb=16`）。checkpoint 中的 `arch` 字段会自动恢复结构。

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

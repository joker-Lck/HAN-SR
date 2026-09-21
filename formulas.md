# SEFNet 数学公式总结

## 1. 网络架构相关公式

### 1.1 全局曝光统计 (GlobalExposureStatistics)

输入特征图 $F \in \mathbb{R}^{B \times 64 \times h \times w}$，全局均值和标准差：

$$\mu = \frac{1}{h \cdot w} \sum_{i=1}^{h} \sum_{j=1}^{w} F_{:, :, i, j}$$

$$\sigma = \sqrt{\frac{1}{h \cdot w} \sum_{i=1}^{h} \sum_{j=1}^{w} (F_{:, :, i, j} - \mu)^2 + \epsilon}, \quad \epsilon = 10^{-6}$$

上下文向量：

$$c = \text{FC}([\mu; \sigma]) \in \mathbb{R}^{B \times d_{ctx}}$$

其中 $[\cdot; \cdot]$ 表示拼接，$d_{ctx} = 64$。

---

### 1.2 正弦位置编码 (Sinusoidal Position Encoding)

用于让网络感知像素空间位置和渐变方向。对高度维度：

$$PE_{2k}(i, j) = \sin\left(\frac{i}{10000^{2k/d}}\right), \quad PE_{2k+1}(i, j) = \cos\left(\frac{i}{10000^{2k/d}}\right)$$

对宽度维度：

$$PE_{d/2+2k}(i, j) = \sin\left(\frac{j}{10000^{2k/d}}\right), \quad PE_{d/2+2k+1}(i, j) = \cos\left(\frac{j}{10000^{2k/d}}\right)$$

其中 $d = d_{feat}/2$，$k = 0, 1, \ldots, d/2-1$。

---

### 1.3 相对曝光编码 (Relative ExposureEncoder)

融合局部特征、位置编码和全局上下文：

$$E_{rel} = f_{head}([f_{local}(x); PE; c_{map}]) \in \mathbb{R}^{B \times d_{feat} \times H \times W}$$

其中 $c_{map} = c.\text{unsqueeze}(2).\text{unsqueeze}(3).\text{expand}(...)$ 是全局上下文广播到每个像素。

---

### 1.4 亮度自适应可变形卷积 (Exposure-Adaptive Deformable Conv)

标准可变形卷积的偏移量只依赖输入特征，本方法同时依赖特征和曝光信息：

$$\Delta p = f_{offset}([x; E_{rel}]) \in \mathbb{R}^{B \times 2K^2 \times H \times W}$$

$$m = \sigma(f_{mask}([x; E_{rel}])) \in \mathbb{R}^{B \times K^2 \times H \times W}$$

$$y(p) = \sum_{k=1}^{K^2} m_k \cdot w_k \cdot x(p + p_k + \Delta p_k)$$

其中 $K$ 是卷积核大小，$m_k$ 是调制掩码（0~1），$\sigma$ 是 sigmoid。

---

### 1.5 门控融合 (Gated Fusion)

标准路径和自适应路径的融合：

$$g = \sigma(f_{gate}([x_{pool}; E_{rel}])) \in \mathbb{R}^{B \times C \times H \times W}$$

$$y = g \odot y_{std} + (1 - g) \odot y_{adapt}$$

其中 $\odot$ 是逐元素乘法，$g \to 1$ 偏向标准路径，$g \to 0$ 偏向自适应路径。

---

### 1.6 残差密集块 (Residual Dense Block, RDB)

5层密集连接：

$$x_1 = \text{Conv}_1(x)$$
$$x_2 = \text{Conv}_2([x; x_1])$$
$$x_3 = \text{Conv}_3([x; x_1; x_2])$$
$$x_4 = \text{Conv}_4([x; x_1; x_2; x_3])$$
$$x_5 = \text{Conv}_5([x; x_1; x_2; x_3; x_4])$$
$$\text{RDB}(x) = 0.2 \cdot x_5 + x$$

其中 $0.2$ 是残差缩放系数（res_scale）。

---

### 1.7 RRDB (Residual in Residual Dense Block)

3个级联RDB + 残差缩放：

$$\text{RRDB}(x) = 0.2 \cdot \text{RDB}_3(\text{RDB}_2(\text{RDB}_1(x))) + x$$

---

### 1.8 全局残差连接 (Global Residual)

浅层特征和深层特征的全局残差：

$$f_{out} = f_{conv\_first}(x) + \text{LR\_conv}(\text{RRDB}_N(\cdots\text{RRDB}_1(f_{conv\_first}(x))))$$

---

### 1.9 PixelShuffle 上采样

将通道数放大 $r^2$ 倍后重排为空间维度：

$$\text{PixelShuffle}: \mathbb{R}^{B \times Cr^2 \times H \times W} \to \mathbb{R}^{B \times C \times rH \times rW}$$

---

## 2. 损失函数公式

### 2.1 L1 像素损失

$$\mathcal{L}_{pixel} = \frac{1}{N} \sum_{i=1}^{N} |I_{SR}^{(i)} - I_{HR}^{(i)}|_1$$

---

### 2.2 VGG 感知损失（激活前特征）

$$\mathcal{L}_{percep} = \frac{1}{C_j H_j W_j} \|\phi_j(I_{SR}) - \phi_j(I_{HR})\|_1$$

其中 $\phi_j$ 是 VGG19 第 $j$ 层（relu5_4之前，第34层）的特征提取器。

---

### 2.3 曝光一致性损失

在VGG特征空间中，曝光差异大的区域降低约束权重：

$$\Delta E = |F_{avg}(\phi(I_{SR})) - F_{avg}(\phi(I_{HR}))|$$

$$w = \exp(-5 \cdot \Delta E)$$

$$\mathcal{L}_{consist} = \frac{1}{C H W} \sum |\phi(I_{SR}) - \phi(I_{HR})| \odot w$$

其中 $F_{avg}$ 是自适应平均池化（将亮度图下采样到VGG特征分辨率）。

---

### 2.4 亮度分布损失

分块统计亮度的均值和标准差，逐块对比：

将图像分为 $16 \times 16$ 的块，计算每块的灰度均值 $\mu_b$ 和标准差 $\sigma_b$：

$$\mathcal{L}_{bright} = \frac{1}{B_n} \sum_{b=1}^{B_n} |\mu_b^{SR} - \mu_b^{HR}| + |\sigma_b^{SR} - \sigma_b^{HR}|$$

其中灰度转换：$Y = 0.299R + 0.587G + 0.114B$（ITU-R BT.601）。

---

### 2.5 总内容损失

$$\mathcal{L}_{content} = \lambda_1 \mathcal{L}_{pixel} + \lambda_2 \mathcal{L}_{percep} + \lambda_3 \mathcal{L}_{consist} + \lambda_4 \mathcal{L}_{bright}$$

默认权重：$\lambda_1 = 1.0, \lambda_2 = 1.0, \lambda_3 = 0.5, \lambda_4 = 0.3$。

---

### 2.6 相对判别器损失 (RaGAN)

判别器判断"真实图比生成图更真实多少"：

$$\mathcal{L}_D = \frac{1}{2} \mathbb{E}\left[\log\left(\sigma(-(D(x_r) - \mathbb{E}[D(x_f)]))\right)\right] + \frac{1}{2} \mathbb{E}\left[\log\left(\sigma(D(x_f) - \mathbb{E}[D(x_r)])\right)\right]$$

---

### 2.7 相对生成器损失 (RaGAN)

$$\mathcal{L}_G^{adv} = \frac{1}{2} \mathbb{E}\left[\log\left(\sigma(-(D(x_r) - \mathbb{E}[D(x_f)]))\right)\right] + \frac{1}{2} \mathbb{E}\left[\log\left(\sigma(D(x_f) - \mathbb{E}[D(x_r)])\right)\right]$$

实际实现用 softplus 替代 log-sigmoid：

$$\text{softplus}(x) = \log(1 + e^x)$$

---

### 2.8 总生成器损失

$$\mathcal{L}_G = \mathcal{L}_{content} + \lambda_{gan} \cdot \mathcal{L}_G^{adv}$$

默认 $\lambda_{gan} = 10^{-3}$。

---

## 3. 训练策略公式

### 3.1 学习率衰减

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T_{decay} \rfloor}$$

默认 $\eta_0 = 2 \times 10^{-4}$，$\gamma = 0.5$，$T_{decay} = 50$。

---

### 3.2 差异化学习率（阶段2）

曝光模块使用更高学习率：

$$\eta_{exposure} = \eta_0, \quad \eta_{main} = 0.5 \cdot \eta_0$$

---

### 3.3 模型插值 (net_interp.py)

在PSNR模型和GAN模型之间线性插值，平衡清晰度和真实感：

$$\theta_{interp} = (1 - \alpha) \cdot \theta_{PSNR} + \alpha \cdot \theta_{GAN}$$

其中 $\alpha \in [0, 1]$，$\alpha = 0$ 纯PSNR，$\alpha = 1$ 纯GAN。

---

## 4. 评价指标公式

### 4.1 PSNR (峰值信噪比)

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1}{\text{MSE}}\right)$$

$$\text{MSE} = \frac{1}{CHW} \sum_{c,i,j} (I_{SR}^{c,i,j} - I_{HR}^{c,i,j})^2$$

值越高越好，典型范围 25~35 dB。

---

### 4.2 SSIM (结构相似度)

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

其中 $C_1 = (0.01)^2$，$C_2 = (0.03)^2$，$\sigma_{xy}$ 是互协方差。值越接近1越好。

---

## 5. 符号说明

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $C$ | 通道数 |
| $H, W$ | 特征图高、宽 |
| $K$ | 卷积核大小 |
| $nf$ | 特征通道数（默认64） |
| $nb$ | RRDB块数量（训练16，测试23） |
| $gc$ | 增长通道数（默认32） |
| $d_{ctx}$ | 曝光上下文维度（默认64） |
| $d_{feat}$ | 曝光特征维度（默认32） |
| $\odot$ | 逐元素乘法 |
| $[\cdot; \cdot]$ | 通道维度拼接 |
| $\phi_j$ | VGG19第$j$层特征 |
| $\sigma$ | sigmoid函数 |

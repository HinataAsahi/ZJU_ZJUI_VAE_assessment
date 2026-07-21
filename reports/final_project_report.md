---
header-includes:
  - \usepackage{graphicx}
---

# Fashion-MNIST VAE 中 KL 正则化的作用分析

## 问题背景与模型

变分自编码器（Variational Autoencoder, VAE）是一类概率生成模型。和普通 autoencoder 只学习一个确定的 latent code 不同，VAE 的 encoder 输出近似后验分布

$$
q_\phi(z\mid x)=\mathcal{N}(\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x))),
$$

并通过 decoder 建模 $p_\theta(x\mid z)$。本项目使用标准正态先验 $p(z)=\mathcal{N}(0,I)$，训练目标为

$$
\mathcal{J}_\beta(x)=
-\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+\beta D_{KL}(q_\phi(z\mid x)\|p(z)).
$$

其中第一项鼓励图像重构，第二项约束 encoder 学到的 latent distribution 接近标准正态先验。`beta` 控制这两种目标之间的权衡：较小的 `beta` 更重视重构，较大的 `beta` 更重视潜空间正则化。

本项目实现了一个用于 28x28 灰度图像的 MLP VAE。模型使用全连接 encoder 和 decoder，隐藏层维度为 `[400, 200]`，latent dim 为 `16`。训练和评估均使用 PyTorch 实现，支持训练、测试集评估、测试图像重构、标准正态先验采样、损失记录和多组实验对比。

## 实验设置

主要实验使用 Fashion-MNIST。所有正式实验保持模型结构和主要训练设置不变，只改变 KL 权重 `beta`。

| setting | value |
|---|---|
| dataset | Fashion-MNIST |
| model | MLP VAE |
| hidden dims | `[400, 200]` |
| latent dim | `16` |
| batch size | `128` |
| epochs | `20` |
| learning rate | `0.001` |
| seed | `42` |
| device | CUDA |

基础任务比较 `beta=1` 的标准 VAE 与 `beta=0` 的 KL 消融模型。开放探索进一步扫描 `beta=0, 0.1, 0.5, 1, 2, 4`，研究 KL 权重变化对重构、KL 和先验采样的影响。

## 基础实验：beta=1 与 beta=0

| run | beta | test reconstruction | test KL | test total |
|---|---:|---:|---:|---:|
| fashion_mnist_beta0 | 0.0 | 214.24 | 297.92 | 214.24 |
| fashion_mnist_beta1 | 1.0 | 228.38 | 12.19 | 240.57 |

不同 `beta` 下的 `test_total` 不适合直接比较，因为它包含不同权重的 KL 项。更合理的比较方式是分别观察 reconstruction loss、KL loss 和生成图像质量。

`beta=0` 的重构损失更低，重构图像也更清晰。这说明去掉 KL 后，模型更接近普通 autoencoder，会优先利用 latent code 记录输入图像细节。但它的 KL 值非常大，说明近似后验明显偏离标准正态先验。

`beta=1` 的重构略更平滑，但 KL 被压到较低范围。从标准正态先验采样时，`beta=1` 生成的图像中有更多可识别的衣物形状；而 `beta=0` 的 prior samples 明显发暗和混乱。这符合 VAE 目标函数的直觉：KL 正则牺牲一部分重构精度，但让从 $N(0,I)$ 采样更可靠。

## 开放探索：beta 系数扫描

开放探索问题是：当 KL 权重 `beta` 从小到大变化时，VAE 的重构质量、KL 大小和先验采样质量如何变化？

| run | beta | test reconstruction | test KL | test total |
|---|---:|---:|---:|---:|
| fashion_mnist_beta0 | 0.0 | 214.24 | 297.92 | 214.24 |
| fashion_mnist_beta0_1 | 0.1 | 215.64 | 37.26 | 219.36 |
| fashion_mnist_beta0_5 | 0.5 | 223.13 | 17.25 | 231.75 |
| fashion_mnist_beta1 | 1.0 | 228.38 | 12.19 | 240.57 |
| fashion_mnist_beta2 | 2.0 | 233.06 | 9.31 | 251.69 |
| fashion_mnist_beta4 | 4.0 | 242.32 | 6.51 | 268.37 |

主要趋势很明确：随着 `beta` 增大，test reconstruction loss 上升，而 test KL 下降。这说明更强的 KL 正则会迫使近似后验更接近标准正态先验，但会降低 latent code 记录输入细节的自由度。

\begin{center}
\includegraphics[width=0.70\linewidth]{reports/figures/beta_sweep_reconstructions.png}

图 1：不同 beta 下的重构结果
\end{center}

从重构图看，`beta=0` 和 `beta=0.1` 的重构最清晰，轮廓和对比度更接近输入。`beta=0.5` 开始出现平滑，但主要类别仍能保留。`beta=1.0`、`2.0` 和 `4.0` 下图像逐渐更平均化，纹理和局部细节减少。

\begin{center}
\includegraphics[width=0.82\linewidth]{reports/figures/beta_sweep_prior_samples.png}

图 2：不同 beta 下的先验采样结果
\end{center}

从 prior samples 看，`beta=0` 的生成结果最差，很多样本只有暗色纹理。`beta=0.1` 已经明显改善采样质量，说明很小的 KL 权重也能显著改善 posterior-prior 匹配。`beta=0.5` 和 `beta=1.0` 在本实验中呈现较好的折中：重构损失没有过高，采样图像中也有较多可识别衣物。`beta=2.0` 和 `beta=4.0` 的采样仍稳定，但结果更保守和模板化。

## 结论、局限与资源说明

本项目的核心结论是：KL 正则化控制了重构质量和生成可靠性之间的权衡。去掉 KL 时，模型能更好地重构输入，但 latent distribution 不再匹配标准正态先验，因此 prior sampling 退化。增大 `beta` 会降低 KL、改善潜空间与先验的匹配，但会提高重构损失并压缩图像细节。对于当前 MLP VAE 和 Fashion-MNIST，`beta=0.5` 到 `beta=1.0` 是比较合理的折中区间。

局限性包括：模型是较简单的 MLP VAE，生成质量有限；实验只使用 Fashion-MNIST，训练轮次为 20；采样质量主要依赖定性观察，没有引入 FID 等复杂指标。后续可以尝试卷积 VAE、更长训练、latent interpolation，或比较不同 latent dim 对潜空间性质的影响。

重要资源和使用方式如下：项目要求 PDF 提供了实验目标和基本公式；Kingma and Welling 的 Auto-Encoding Variational Bayes、Rezende et al. 的 stochastic backpropagation 论文、Higgins et al. 的 beta-VAE 论文用于理解 VAE、重参数化和 beta 权重思想；PyTorch 与 torchvision 用于模型实现和 Fashion-MNIST 数据加载；AI 工具用于辅助解释概念、生成学习笔记、整理报告草稿和检查代码，但实验代码、运行结果和报告结论均基于本仓库实现与实际输出。

## 参考文献

1. D. P. Kingma and M. Welling. *Auto-Encoding Variational Bayes*, 2014.
2. D. J. Rezende, S. Mohamed, and D. Wierstra. *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*, 2014.
3. I. Higgins et al. *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, 2017.

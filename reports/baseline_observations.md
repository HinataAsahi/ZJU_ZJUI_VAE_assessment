# Fashion-MNIST 基线实验观察

本文记录 MLP VAE 在 Fashion-MNIST 上的基础对照实验。目标是比较标准 VAE (`beta=1`) 与去除 KL 正则的消融设置 (`beta=0`)，观察 KL 项对重构质量、潜空间约束和先验采样质量的影响。

## 实验设置

两组实验保持主要设置一致，只改变 `beta`：

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

运行配置：

- `configs/fashion_mnist_beta1.yaml`
- `configs/fashion_mnist_beta0.yaml`

## 指标汇总

指标表由 `scripts/compare_runs.py` 从 `outputs/*/evaluation.json` 和 `outputs/*/metrics.json` 读取生成。

| run | dataset | beta | epochs | test reconstruction | test KL | test total |
|---|---|---:|---:|---:|---:|---:|
| fashion_mnist_beta1 | fashion_mnist | 1.0 | 20 | 228.38 | 12.19 | 240.57 |
| fashion_mnist_beta0 | fashion_mnist | 0.0 | 20 | 214.24 | 297.92 | 214.24 |

不同 `beta` 下的 `test_total` 不适合直接比较，因为 `test_total = reconstruction + beta * KL`。更有意义的是分别比较 reconstruction loss、KL loss 和生成图像质量。

## 重构结果观察

`beta=0` 的重构图比 `beta=1` 更清晰，轮廓和局部对比度更接近输入图像。这与指标一致：`beta=0` 的 test reconstruction loss 更低。

`beta=1` 的重构图仍能保留主要类别和大体形状，但细节更平滑。这是合理现象，因为标准 VAE 在优化重构项的同时，还需要让近似后验接近标准正态先验。

相关文件：

- `outputs/fashion_mnist_beta1/figures/reconstructions.png`
- `outputs/fashion_mnist_beta0/figures/reconstructions.png`

## 先验采样观察

从标准正态先验 `N(0, I)` 采样时，`beta=1` 生成的图像中有较多可识别的衣物轮廓，例如上衣、裤子、鞋子和包。虽然细节仍然有限，但整体更接近 Fashion-MNIST 的数据分布。

`beta=0` 的 prior samples 明显更暗、更混乱，很多样本只能看到模糊纹理或不稳定轮廓。这说明去除 KL 后，encoder 学到的潜变量分布不再被约束到标准正态附近。训练时 decoder 主要见到的是 encoder 产生的 latent，而不是标准正态区域中的随机点，所以从 `N(0, I)` 直接采样会退化。

相关文件：

- `outputs/fashion_mnist_beta1/figures/prior_samples.png`
- `outputs/fashion_mnist_beta0/figures/prior_samples.png`

## 损失曲线观察

`beta=1` 中，test KL 维持在约 `12` 附近，说明 KL 正则持续约束 latent distribution。重构损失从前几轮快速下降，后期趋于平缓。

`beta=0` 中，重构损失下降到更低水平，但 KL 从第 1 轮的约 `193.21` 增长到第 20 轮的约 `297.92`。由于 KL 不进入优化目标，模型没有动力保持 posterior 接近 `N(0, I)`，因此 KL 膨胀是预期现象。

相关文件：

- `outputs/fashion_mnist_beta1/figures/loss_curves.png`
- `outputs/fashion_mnist_beta0/figures/loss_curves.png`

## 结果解释

这组实验体现了 VAE 的核心 trade-off：

- `beta=0` 更像普通 autoencoder，主要追求重构输入，因此 reconstruction loss 更低。
- `beta=1` 是标准 VAE 目标，在重构之外加入 KL 正则，让 latent space 更接近标准正态先验。
- KL 正则会牺牲一部分重构精度，但能提高从先验采样生成图像的可靠性。

因此，`beta=0` 并不是更好的生成模型。它在重构任务上更强，但在 generative sampling 上更弱。

## 局限性与下一步

当前基础实验只比较了 `beta=1` 和 `beta=0` 两个端点，足以说明 KL 项的作用，但还不能完整刻画不同正则强度之间的连续变化。

后续可以使用 `scripts/compare_runs.py` 对更多实验进行自动汇总，例如：

```bash
PYTHONPATH=src python scripts/compare_runs.py \
  outputs/fashion_mnist_beta0 \
  outputs/fashion_mnist_beta1 \
  --title "Fashion-MNIST beta comparison"
```

如果继续开放探索，一个自然方向是增加更多 `beta` 值，例如 `0.1`、`0.5`、`2.0` 或 `4.0`，观察 reconstruction loss、KL loss 和 prior samples 如何随 KL 权重变化。

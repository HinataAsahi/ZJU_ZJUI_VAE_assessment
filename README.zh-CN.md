# ZJU ZJUI VAE Assessment 中文说明

本仓库是面向 PIL Lab 2026 暑期小项目的 VAE 基础实现。当前目标不是直接追求复杂模型，而是先建立一条清晰的学习和实验路线：理解 VAE 的基本思想，完成一个可运行、可测试、可复现实验的 PyTorch 基线，然后再基于这个基线做开放探索。

## 项目内容

当前代码实现了一个用于 28x28 灰度图像的 MLP VAE，适合 MNIST 或 Fashion-MNIST 这类入门数据集。它支持：

- 训练和测试集评估；
- 图像重建可视化；
- 从标准正态先验 `N(0, I)` 采样生成图像；
- 记录总损失、重建损失和 KL 损失；
- 对比 `beta=1` 标准 VAE 与 `beta=0` 去除 KL 约束后的效果。

## 推荐学习顺序

先打开学习 notebook：

```text
notebooks/01_vae_basics.ipynb
```

建议按顺序阅读和运行。这个 notebook 从零解释：

1. VAE 和普通 Autoencoder 的区别；
2. 为什么 encoder 输出的是 `mu` 和 `logvar`；
3. 为什么要使用 `z = mu + std * eps` 的重参数化技巧；
4. 为什么 VAE 的损失是重建项加上 `beta * KL`；
5. 为什么 `beta=0` 往往重建更好，但从随机先验采样的图像会更差。

从仓库根目录或 `notebooks/` 目录启动 Jupyter 都可以。notebook 的第一个代码单元会自动定位仓库根目录并加入 `src/` 到 Python 导入路径，因此不需要额外设置 `PYTHONPATH`。

## 环境准备

如果需要新建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你的笔记本已经装好了 PyTorch，尤其是已经有 CUDA 版 PyTorch，可以保留现有 PyTorch，只补装 `requirements.txt` 中缺失的依赖。不要为了这个项目强行重装 PyTorch。

## 本地快速验证

烟雾测试配置使用 `FakeData`，不会下载 MNIST 或 Fashion-MNIST，适合先确认代码链路能跑通：

```bash
PYTHONPATH=src python scripts/train.py --config configs/mnist_smoke.yaml
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/mnist_smoke --device cpu
```

运行后会生成：

```text
outputs/mnist_smoke/checkpoint.pt
outputs/mnist_smoke/config.json
outputs/mnist_smoke/metrics.json
outputs/mnist_smoke/evaluation.json
outputs/mnist_smoke/figures/reconstructions.png
outputs/mnist_smoke/figures/prior_samples.png
outputs/mnist_smoke/figures/loss_curves.png
```

这些文件属于本地实验输出，不应该提交到 GitHub。

## Fashion-MNIST 基础实验

有 4060 笔记本时，建议优先在 GPU 上跑正式实验。脚本里的 `--device auto` 会自动选择 CUDA；也可以显式写成 `--device cuda`。

标准 VAE：

```bash
PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta1.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta1 --device auto
```

去除 KL 的对照实验：

```bash
PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta0.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta0 --device auto
```

## β 系数扫描开放探索

基础实验之后，可以继续运行一组更细的 `beta` 扫描，用来观察 KL 权重如何影响重构误差、KL 大小和先验采样质量。当前已提供：

```text
configs/fashion_mnist_beta0.yaml
configs/fashion_mnist_beta0_1.yaml
configs/fashion_mnist_beta0_5.yaml
configs/fashion_mnist_beta1.yaml
configs/fashion_mnist_beta2.yaml
configs/fashion_mnist_beta4.yaml
```

新增实验命令：

```bash
PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta0_1.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta0_1 --device auto

PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta0_5.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta0_5 --device auto

PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta2.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta2 --device auto

PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta4.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta4 --device auto
```

复制结果回来后，可用对比脚本汇总所有 run：

```bash
PYTHONPATH=src python scripts/compare_runs.py \
  outputs/fashion_mnist_beta0 \
  outputs/fashion_mnist_beta0_1 \
  outputs/fashion_mnist_beta0_5 \
  outputs/fashion_mnist_beta1 \
  outputs/fashion_mnist_beta2 \
  outputs/fashion_mnist_beta4 \
  --title "Fashion-MNIST beta sweep"
```

如果本机 CUDA 环境没有被识别，可以先在 Python 中检查：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 测试

开发或修改代码后运行：

```bash
PYTHONPATH=src pytest -v
```

当前测试覆盖了配置读取、数据加载、VAE 前向传播、损失函数、训练/评估 smoke test、可视化输出，以及学习 notebook 的基本可执行性。

## 实验结果应该怎么比较

报告中建议重点比较：

- test reconstruction loss；
- test KL loss；
- reconstruction grids；
- prior sampling grids；
- loss curves。

不要直接比较不同 `beta` 设置下的 `test_total`，因为 `test_total` 本身包含不同权重的 KL 项。更合理的比较方式是分别观察重建误差、KL 大小、重建图像质量、从先验随机采样出来的图像质量。

核心分析方向是：去掉 KL 后，模型通常会更专注于重建训练图像，但潜变量分布不再被约束到标准正态附近。因此从 `N(0, I)` 随机采样时，生成图像的可靠性通常会下降。

## 当前阶段定位

这个仓库目前处在“基础学习 + 基线实现”阶段。推荐推进顺序是：

1. 学完 `notebooks/01_vae_basics.ipynb`；
2. 跑通 `mnist_smoke.yaml`；
3. 在 4060 笔记本上运行两个 Fashion-MNIST 基础配置；
4. 对比 `beta=1` 和 `beta=0` 的指标与图片；
5. 运行 `beta=0.1, 0.5, 2, 4` 的开放探索配置；
6. 汇总并分析不同 KL 权重下的趋势。

当前开放探索主题是 β 系数扫描。重点不是追求最漂亮的图片，而是解释 reconstruction loss、KL loss 和 prior samples 随 KL 权重变化的趋势。

## Git 和文件规范

仓库会忽略本地数据、训练输出、checkpoint、缓存文件和内部规划材料。通常只提交源码、配置、测试、README 和学习 notebook。

不要提交：

- `outputs/`；
- `data/`；
- `*.pt`、`*.pth`；
- 本地缓存目录；
- 课程要求 PDF；
- 内部 spec 或计划文档。

这样可以保证 GitHub 仓库保持轻量、可复现，并且不会把本地实验产物或课程原始材料混入公开历史。

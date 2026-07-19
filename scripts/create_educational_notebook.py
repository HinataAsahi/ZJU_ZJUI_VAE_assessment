from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        "# 01 VAE Basics：从零理解变分自编码器\n\n"
        "这个 notebook 用一个小型 MLP VAE 串起直觉、公式、代码和实验现象。目标不是先追求漂亮图片，而是能解释每个张量为什么存在、每一项 loss 在约束什么，以及图像结果应该怎样读。\n"
    ),
    markdown(
        "## 学习路线\n\n"
        "我们会依次回答四类问题：为什么需要 VAE；概率符号怎样对应代码；训练循环如何工作；重构与生成结果怎样解释。默认路径完全离线，所有代码单元按顺序运行即可。\n"
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "for candidate in (Path.cwd(), *Path.cwd().parents):\n"
        "    if (candidate / 'src' / 'vae_project').is_dir():\n"
        "        project_root = candidate\n"
        "        break\n"
        "else:\n"
        "    raise FileNotFoundError('Could not locate the repository root containing src/vae_project')\n"
        "src_path = str(project_root / 'src')\n"
        "if src_path not in sys.path:\n"
        "    sys.path.insert(0, src_path)\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import torch\n"
        "from vae_project.config import validate_config\n"
        "from vae_project.data import get_dataloaders\n"
        "from vae_project.losses import kl_divergence_standard_normal, vae_loss\n"
        "from vae_project.models import MLPVAE\n"
        "from vae_project.utils import select_device, set_seed\n"
        "from vae_project.visualization import save_prior_samples, save_reconstruction_grid\n"
    ),
    markdown(
        "## 从 Autoencoder 到 VAE：为什么普通 AE 不能自然生成\n\n"
        "普通 Autoencoder（AE）把输入 `x` 编码成一个确定的向量，再从这个向量重构 `x`。训练只要求训练样本自己的编码能被 decoder 还原，没有要求不同样本的编码连续、密集，或服从某个已知分布。\n"
    ),
    markdown(
        "AE 的 latent space 里可能有很多 decoder 从未见过的空洞。训练后随便采一个向量，可能正好落在空洞中，因此 decoder 不知道如何把它变成合理图像。VAE 增加分布约束，让各样本的 latent distribution 靠近一个可直接采样的先验。\n\n"
        "一句话区分：AE 主要学习 `x -> z -> x_hat`；VAE 还要让“从哪里采 `z`”这件事有定义。\n"
    ),
    markdown(
        "## 概率建模直觉：`z`、`p(z)`、`q(z|x)`、`p(x|z)` 分别是什么\n\n"
        "VAE 把生成过程写成：先从 `p(z)` 采潜变量，再由 `p(x|z)` 生成图像。训练时只有图像 `x`，真实后验 `p(z|x)` 很难直接计算，所以用 encoder 表示的 `q(z|x)` 去近似它。\n"
    ),
    markdown(
        "| 符号 | 直觉 | 在本项目中的对应物 |\n"
        "| --- | --- | --- |\n"
        "| `z` | 描述图像隐藏因素的 latent vector | `output['z']` |\n"
        "| `p(z)` | 生成时使用的先验，通常是 `N(0, I)` | `torch.randn(...)` |\n"
        "| `q(z|x)` | encoder 给定图像后预测的近似后验 | `mu` 和 `logvar` 定义的高斯分布 |\n"
        "| `p(x|z)` | decoder 给定 `z` 后对像素的分布 | decoder 输出的 Bernoulli logits |\n"
    ),
    markdown(
        "这里的 `q` 和 `p` 不是四个独立网络。Encoder 负责给出 `q(z|x)` 的参数，decoder 负责给出 `p(x|z)` 的参数，而 `p(z)=N(0,I)` 是我们预先选择的简单分布。\n"
    ),
    markdown(
        "## 如何从 FakeData 切到 MNIST，再切到 Fashion-MNIST\n\n"
        "默认的 `dataset: 'fake'` 使用 FakeData。`fake`（FakeData）只用于离线 smoke/学习，可以验证 shape、loss 和训练路径，但随机图案不会产生有语义的生成效果。\n\n"
        "`mnist`（MNIST）用于真实数据上的代码调试，容易观察数字轮廓。`fashion_mnist`（Fashion-MNIST）是作业要求的主实验/正式实验数据集，最终的 `beta=1` 与 `beta=0` 比较必须在它上面完成。真实数据需要事先下载或把 `download` 显式改为 `True`；本 notebook 默认不触发下载。\n"
    ),
    markdown(
        "### 先使用离线小配置\n\n"
        "限制训练和测试样本数，只跑少量 batch。固定随机种子并使用 CPU，使教学结果更容易复现。图像经 `ToTensor()` 后位于 `[0, 1]`。\n"
    ),
    code(
        "config = validate_config({\n"
        "    'run_name': 'notebook_smoke',\n"
        "    'dataset': 'fake',\n"
        "    'data_dir': 'data',\n"
        "    'output_dir': 'outputs/notebook_smoke',\n"
        "    'batch_size': 32,\n"
        "    'epochs': 1,\n"
        "    'learning_rate': 1e-3,\n"
        "    'seed': 42,\n"
        "    'device': 'cpu',\n"
        "    'beta': 1.0,\n"
        "    'latent_dim': 8,\n"
        "    'hidden_dims': [128, 64],\n"
        "    'train_limit': 128,\n"
        "    'test_limit': 64,\n"
        "    'num_workers': 0,\n"
        "    'download': False,\n"
        "    'sample_count': 16,\n"
        "})\n"
        "set_seed(config['seed'])\n"
        "device = select_device(config['device'])\n"
        "train_loader, test_loader = get_dataloaders(config)\n"
        "device, len(train_loader), len(test_loader)\n"
    ),
    markdown(
        "### 检查一个 batch 的 shape 和数值范围\n\n"
        "模型期望输入 shape 为 `[batch, channel, height, width]`。在计算 BCE 前，还应确认 target 像素确实位于 `[0, 1]`。\n"
    ),
    code(
        "images, labels = next(iter(train_loader))\n"
        "batch_summary = {\n"
        "    'images_shape': tuple(images.shape),\n"
        "    'labels_shape': tuple(labels.shape),\n"
        "    'pixel_min': float(images.min()),\n"
        "    'pixel_max': float(images.max()),\n"
        "    'dtype': str(images.dtype),\n"
        "}\n"
        "batch_summary\n"
    ),
    markdown(
        "### 先看输入图像\n\n"
        "FakeData 的图像是用于打通流程的随机模式，不要把它误读成模型失败。切换到 MNIST 或 Fashion-MNIST 后，才应期待数字或服饰轮廓。\n"
    ),
    code(
        "fig, axes = plt.subplots(1, 6, figsize=(9, 2))\n"
        "for index, axis in enumerate(axes):\n"
        "    axis.imshow(images[index, 0], cmap='gray', vmin=0, vmax=1)\n"
        "    axis.set_title(f'label={int(labels[index])}')\n"
        "    axis.axis('off')\n"
        "plt.tight_layout()\n"
    ),
    markdown(
        "## Encoder 为什么输出 `mu` 和 `logvar`\n\n"
        "VAE 不把一张图编码成单点，而是编码成对角高斯分布 `q(z|x)=N(mu, diag(sigma^2))`。`mu` 表示分布中心，方差表示对每个 latent dimension 的不确定性。\n"
    ),
    markdown(
        "网络输出 `logvar = log(sigma^2)` 而不是直接输出 `sigma`。这样输出可以是任意实数，不需要额外保证方差为正；需要标准差时使用 `std = exp(0.5 * logvar)`，结果自然大于零。\n"
    ),
    code(
        "model = MLPVAE(\n"
        "    input_shape=(1, 28, 28),\n"
        "    hidden_dims=config['hidden_dims'],\n"
        "    latent_dim=config['latent_dim'],\n"
        ").to(device)\n"
        "images = images.to(device)\n"
        "mu, logvar = model.encode(images)\n"
        "mu.shape, logvar.shape\n"
    ),
    markdown(
        "### 查看 `mu`、`logvar`、`std` 和一次采样的 `z`\n\n"
        "先只看第一张图和前四个 latent dimensions。初始网络尚未训练，数值本身没有语义；重点是四者的计算关系和相同 shape。\n"
    ),
    code(
        "std = torch.exp(0.5 * logvar)\n"
        "latent_generator = torch.Generator(device='cpu').manual_seed(config['seed'])\n"
        "z = model.reparameterize(mu, logvar, sample=True, generator=latent_generator)\n"
        "{\n"
        "    'mu': mu[0, :4].detach().cpu(),\n"
        "    'logvar': logvar[0, :4].detach().cpu(),\n"
        "    'std': std[0, :4].detach().cpu(),\n"
        "    'sampled_z': z[0, :4].detach().cpu(),\n"
        "}\n"
    ),
    markdown(
        "## 重参数化技巧为什么能让采样参与反向传播\n\n"
        "如果把 `z ~ N(mu, sigma^2)` 当作一个黑盒随机操作，梯度无法说明随机样本怎样随 `mu`、`sigma` 改变。重参数化把它改写为 `epsilon ~ N(0,I)` 与 `z = mu + std * epsilon`。\n"
    ),
    markdown(
        "随机性被移到与参数无关的 `epsilon`；给定一次 `epsilon` 后，`z` 是 `mu` 和 `std` 的普通可微函数。因此 reconstruction loss 的梯度可以沿 `decoder -> z -> mu/std -> encoder` 回传。采样仍然存在，只是采样源不再包含待学习参数。\n"
    ),
    code(
        "manual_generator = torch.Generator(device='cpu').manual_seed(7)\n"
        "epsilon = torch.randn(std.shape, generator=manual_generator, device='cpu').to(device)\n"
        "manual_z = mu + std * epsilon\n"
        "manual_z.requires_grad, manual_z.grad_fn\n"
    ),
    markdown(
        "## ELBO 直觉：重构项和 KL 项各自在约束什么\n\n"
        "当 `beta=1` 时，`reconstruction + KL` 是负 ELBO：这里的 reconstruction 是负对数似然（本项目用 BCE 计算），所以最小化它等价于最大化 ELBO。两个项会拉扯：重构项希望每个 `z` 保留足够多的输入细节；KL 项希望每个 `q(z|x)` 不要偏离统一的先验 `N(0,I)` 太远。\n\n"
        "当 `beta != 1`（包括本作业的 `beta=0`）时，`reconstruction + beta * KL` 是 beta-VAE 加权目标 / 消融目标，不是标准 ELBO。它仍然是有用的训练目标，但其目的变为刻意改变重构与先验对齐之间的权衡。\n"
    ),
    markdown(
        "- **重构项**约束 decoder 能否由当前 `z` 解释输入，回答“信息够不够”。\n"
        "- **KL 项**约束 encoder 的分布能否与生成时的先验对齐，回答“潜空间能不能直接采”。\n"
        "- **`beta`**控制两者权衡；本作业用 `beta=1` 与 `beta=0` 做最直接的消融比较。\n"
    ),
    code(
        "output = model(images, sample=True)\n"
        "losses = vae_loss(\n"
        "    output['recon_logits'], images, output['mu'], output['logvar'], beta=config['beta']\n"
        ")\n"
        "{name: float(value.detach().cpu()) for name, value in losses.items()}\n"
    ),
    markdown(
        "## KL 公式和代码如何对应\n\n"
        "对角高斯 `q(z|x)=N(mu, diag(sigma^2))` 与标准正态先验的 KL 为：\n\n"
        "`KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))`。\n\n"
        "代码先对 latent dimensions 求和得到每个样本的 KL，再对 batch 求平均。\n"
    ),
    markdown(
        "公式中的 `1` 是常数；`logvar` 对应 `log(sigma^2)`；`mu.pow(2)` 惩罚均值偏离 0；`logvar.exp()` 还原方差并惩罚它偏离 1。若 `mu=0` 且 `logvar=0`，则方差为 1，每一维括号内恰好为 0。\n"
    ),
    code(
        "standard_mu = torch.zeros(1, 4)\n"
        "standard_logvar = torch.zeros(1, 4)\n"
        "shifted_mu = torch.ones(1, 4)\n"
        "standard_kl = kl_divergence_standard_normal(standard_mu, standard_logvar)\n"
        "shifted_kl = kl_divergence_standard_normal(shifted_mu, standard_logvar)\n"
        "{'standard_normal_KL': float(standard_kl), 'shifted_posterior_KL': float(shifted_kl)}\n"
    ),
    markdown(
        "标准正态 posterior 的 KL 应为 0；仅把四维均值都移到 1 后，KL 应变为正数。这个小例子把“KL 衡量与先验的距离”变成可检查的数值。\n"
    ),
    markdown(
        "## BCE reconstruction loss、logits、sigmoid 的关系\n\n"
        "Decoder 输出的是 raw logits，不是已经位于 `[0,1]` 的像素。`sigmoid(logits)` 才能得到 Bernoulli 概率或可显示的灰度值。训练使用 `binary_cross_entropy_with_logits`，它在一个数值稳定的操作里组合 sigmoid 和 BCE。\n"
    ),
    markdown(
        "因此有两个容易混淆但必须分开的规则：把 `recon_logits` 直接交给 loss；显示或保存图像前再调用 sigmoid。若训练前手动 sigmoid，再使用 `BCEWithLogits`，就相当于重复做了一次 sigmoid。\n"
    ),
    code(
        "example_logits = torch.tensor([-4.0, 0.0, 4.0])\n"
        "example_probabilities = torch.sigmoid(example_logits)\n"
        "torch.stack([example_logits, example_probabilities], dim=1)\n"
    ),
    markdown(
        "## 训练循环逐步拆解：forward、loss、backward、optimizer\n\n"
        "一个 batch 的学习过程固定由四步组成。理解顺序比记住 API 更重要，因为漏掉清梯度、反向传播或参数更新都会让训练行为改变。\n"
    ),
    markdown(
        "### 1. forward\n\n"
        "`output = model(batch_images)` 依次执行 encoder、重参数化和 decoder，返回 `recon_logits`、`mu`、`logvar`、`z`。调用 `model.train()` 后，默认会采样 latent。\n"
    ),
    markdown(
        "### 2. loss\n\n"
        "`vae_loss(...)` 用 target 图像和 logits 算 reconstruction，再用 `mu/logvar` 算 KL，最后得到 `total = reconstruction + beta * KL`。只有 `total` 用于反向传播，另外两项用于诊断。\n"
    ),
    markdown(
        "### 3. backward\n\n"
        "先用 `optimizer.zero_grad(set_to_none=True)` 清除上一 batch 的梯度，再调用 `losses['total'].backward()`。Autograd 会沿 decoder、重参数化表达式和 encoder 计算梯度。\n"
    ),
    markdown(
        "### 4. optimizer\n\n"
        "`optimizer.step()` 根据刚算出的梯度更新参数。它必须发生在 backward 之后；参数更新完成后，才进入下一个 batch。\n"
    ),
    markdown(
        "### 在 FakeData 上运行一个 tiny training loop\n\n"
        "下面只训练三个 batch，目的是验证完整学习路径，而不是获得有意义的图像质量。记录 total、reconstruction 和 KL，方便确认三项都是有限数值。\n"
    ),
    code(
        "set_seed(config['seed'])\n"
        "optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])\n"
        "tiny_history = []\n"
        "model.train()\n"
        "for step, (batch_images, _batch_labels) in enumerate(train_loader, start=1):\n"
        "    batch_images = batch_images.to(device)\n"
        "    optimizer.zero_grad(set_to_none=True)\n"
        "    output = model(batch_images)\n"
        "    losses = vae_loss(\n"
        "        output['recon_logits'], batch_images, output['mu'], output['logvar'], beta=config['beta']\n"
        "    )\n"
        "    losses['total'].backward()\n"
        "    optimizer.step()\n"
        "    tiny_history.append({name: float(value.detach().cpu()) for name, value in losses.items()})\n"
        "    if step == 3:\n"
        "        break\n"
        "tiny_history\n"
    ),
    markdown(
        "训练时优先检查：loss 是否有限、是否整体下降、KL 是否长期接近 0、reconstruction 与 KL 的量级是否严重失衡。三个 FakeData batch 只能做 smoke test，不能用于判断收敛。\n"
    ),
    markdown(
        "## 重构图怎么看\n\n"
        "重构图比较“给模型一张图后，它保留了多少输入信息”。本项目保存的网格上排是原图，下排是对应重构；位置一一对应。先看大轮廓，再看类别特征和细节，不要只凭是否清晰下结论。\n"
    ),
    markdown(
        "重构模糊可能来自训练不足、latent dimension 太小、KL 约束太强，或 BCE 倾向输出平均化像素。重构好只说明 encoder-decoder 路径有效，不自动证明先验采样有效。\n"
    ),
    code(
        "figures_dir = Path(config['output_dir']) / 'figures'\n"
        "reconstruction_path = figures_dir / 'notebook_reconstructions.png'\n"
        "save_reconstruction_grid(model, test_loader, device, reconstruction_path, max_images=8)\n"
        "plt.figure(figsize=(10, 3))\n"
        "plt.imshow(plt.imread(reconstruction_path))\n"
        "plt.axis('off')\n"
        "plt.tight_layout()\n"
    ),
    markdown(
        "## 先验采样图怎么看\n\n"
        "先验采样不提供输入图像，而是直接取 `z ~ N(0,I)` 再 decode。它检验的是训练得到的 latent space 是否与生成时使用的先验对齐，所以比重构更直接地反映 VAE 的生成能力。\n"
    ),
    markdown(
        "观察时看三点：单张样本是否像数据分布；不同格子是否有多样性；是否存在大量噪声或几乎相同的输出。FakeData 没有稳定语义，正式判断必须使用训练充分的 Fashion-MNIST 模型。\n"
    ),
    code(
        "prior_path = figures_dir / 'notebook_prior_samples.png'\n"
        "prior_generator = torch.Generator(device='cpu').manual_seed(config['seed'])\n"
        "prior_latents = torch.randn(16, model.latent_dim, generator=prior_generator)\n"
        "save_prior_samples(model, device, prior_path, latents=prior_latents)\n"
        "plt.figure(figsize=(5, 5))\n"
        "plt.imshow(plt.imread(prior_path))\n"
        "plt.axis('off')\n"
        "plt.tight_layout()\n"
    ),
    markdown(
        "## `beta=1` vs `beta=0` 运行前应该预期什么\n\n"
        "`beta=1` 使用标准 VAE 的负 ELBO 目标；`beta=0` 完全移除 KL 对 total loss 的影响，因此是消融目标而不是标准 ELBO。公平比较时只改变 beta，保持数据、网络、seed、epoch、batch size 和学习率一致。\n"
    ),
    markdown(
        "| 观察项 | `beta=1` 的预期 | `beta=0` 的预期 |\n"
        "| --- | --- | --- |\n"
        "| reconstruction | 可能略差 | 可能更好 |\n"
        "| KL | 被纳入优化，通常受控 | 仍可计算，但没有梯度权重，可能漂移 |\n"
        "| prior samples | 更可能形成可识别且连续的样本 | 可能差，即使重构很好 |\n"
        "| latent space | 更接近 `N(0,I)` | 只服务于重构，没有先验对齐保证 |\n\n"
        "这些是运行前的假设，不是必须出现的结论；最终要用 loss 曲线、重构网格和先验采样网格共同验证。\n"
    ),
    markdown(
        "### 数据集切换配置\n\n"
        "下面只展示配置差异，不创建真实数据 loader，因此不会下载。已有本地数据时可保持 `download=False`；需要首次下载时，由你主动改为 `True`。\n"
    ),
    code(
        "dataset_modes = {\n"
        "    'offline_learning': {'dataset': 'fake', 'download': False},\n"
        "    'real_data_debugging': {'dataset': 'mnist', 'download': False},\n"
        "    'formal_assignment': {'dataset': 'fashion_mnist', 'download': False},\n"
        "}\n"
        "dataset_modes\n"
    ),
    markdown(
        "## 常见问题：posterior collapse、KL 太小/太大、采样差但重构好\n\n"
        "**Posterior collapse** 指 encoder 的 `q(z|x)` 几乎退化为与输入无关的先验，decoder 很少使用 `z`。典型信号是 KL 长期接近 0、不同输入的 `mu` 很相似，以及改变 `z` 对输出影响很小。\n"
    ),
    markdown(
        "- **KL 太小**：先确认 decoder 是否忽略 `z`、beta 是否过强、训练是否充分。KL 小不总是好事。\n"
        "- **KL 太大**：posterior 与先验偏离明显；检查 loss reduction、输入范围、学习率和 beta。\n"
        "- **采样差但重构好**：常见于 `beta=0` 或先验对齐不足。重构使用 `q(z|x)` 附近的点，而生成使用 `p(z)` 的点，两者可能不在同一区域。\n"
        "- **所有样本很像**：检查训练不足、模型容量、数据问题，也要结合 KL 判断是否 collapse。\n"
    ),
    markdown(
        "## 读完后你应该能回答的问题\n\n"
        "1. 普通 AE 的 latent space 为什么不能保证随机采样有效？\n"
        "2. `p(z)`、`q(z|x)` 和 `p(x|z)` 分别由什么定义？\n"
        "3. 为什么 encoder 输出 `logvar`，而不是直接输出标准差？\n"
        "4. `z = mu + std * epsilon` 为什么保留了到 encoder 的梯度路径？\n"
        "5. KL 为 0 在数学上意味着什么？训练中 KL 长期接近 0 又可能意味着什么？\n"
        "6. 为什么 loss 接收 logits，而可视化必须使用 sigmoid 后的值？\n"
        "7. 一个训练 batch 中 forward、loss、backward、optimizer 的顺序是什么？\n"
        "8. 重构好但先验采样差，说明潜空间可能有什么问题？\n"
        "9. 比较 `beta=1` 与 `beta=0` 时，哪些设置必须保持不变？\n"
        "10. FakeData、MNIST、Fashion-MNIST 在本项目中分别承担什么角色？\n"
    ),
    markdown(
        "### 答辩前最后自检\n\n"
        "不要只背公式。尝试用一张图的路径讲完整故事：`x -> mu/logvar -> z -> logits -> reconstruction loss`，再解释 KL 怎样让这条重构路径与 `z ~ N(0,I)` 的生成路径连接起来。最后用 `beta=0` 的反例说明为什么“重构好”不等于“会生成”。\n"
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def write_notebook(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    output_path = Path("notebooks/01_vae_basics.ipynb")
    write_notebook(output_path)
    print(output_path)


if __name__ == "__main__":
    main()

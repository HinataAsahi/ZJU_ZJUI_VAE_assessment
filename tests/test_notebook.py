import json
import os
import subprocess
import sys
from pathlib import Path


def test_educational_notebook_exists_and_has_required_sections():
    notebook_path = Path("notebooks/01_vae_basics.ipynb")
    assert notebook_path.exists()

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 45
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])

    required_phrases = [
        "从 Autoencoder 到 VAE：为什么普通 AE 不能自然生成",
        "概率建模直觉：`z`、`p(z)`、`q(z|x)`、`p(x|z)` 分别是什么",
        "Encoder 为什么输出 `mu` 和 `logvar`",
        "重参数化技巧为什么能让采样参与反向传播",
        "ELBO 直觉：重构项和 KL 项各自在约束什么",
        "KL 公式和代码如何对应",
        "BCE reconstruction loss、logits、sigmoid 的关系",
        "训练循环逐步拆解：forward、loss、backward、optimizer",
        "重构图怎么看",
        "先验采样图怎么看",
        "`beta=1` vs `beta=0` 运行前应该预期什么",
        "如何从 FakeData 切到 MNIST，再切到 Fashion-MNIST",
        "常见问题：posterior collapse、KL 太小/太大、采样差但重构好",
        "读完后你应该能回答的问题",
        "`fake`（FakeData）只用于离线 smoke/学习",
        "`mnist`（MNIST）用于真实数据上的代码调试",
        "`fashion_mnist`（Fashion-MNIST）是作业要求的主实验/正式实验数据集",
    ]
    for phrase in required_phrases:
        assert phrase in joined_source


def test_notebook_import_cell_works_without_pythonpath(tmp_path):
    notebook_path = Path("notebooks/01_vae_basics.ipynb")
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    import_cell = next(cell for cell in data["cells"] if cell["cell_type"] == "code")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")

    result = subprocess.run(
        [sys.executable, "-c", "".join(import_cell["source"])],
        cwd=notebook_path.parent,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

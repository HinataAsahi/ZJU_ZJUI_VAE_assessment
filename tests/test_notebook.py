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
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])

    required_phrases = [
        "VAE 要解决什么问题",
        "Encoder 输出 mu 和 logvar",
        "重参数化技巧",
        "ELBO 损失",
        "训练一个小型 VAE",
        "重构测试图像",
        "从先验分布采样",
        "beta=1 和 beta=0",
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

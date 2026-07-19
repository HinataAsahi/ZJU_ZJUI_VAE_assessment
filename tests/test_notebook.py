import json
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
    ]
    for phrase in required_phrases:
        assert phrase in joined_source

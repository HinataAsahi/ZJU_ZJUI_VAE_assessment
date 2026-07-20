import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks/01_vae_basics.ipynb"


def load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_unescaped_pipes(line):
    count = 0
    escaped = False
    for character in line:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "|":
            count += 1
    return count


def source_cells(notebook):
    return [
        {"cell_type": cell["cell_type"], "source": cell.get("source", [])}
        for cell in notebook["cells"]
    ]


def test_educational_notebook_exists_and_has_required_sections():
    assert NOTEBOOK_PATH.exists()

    data = load_notebook(NOTEBOOK_PATH)
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 45
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])

    required_phrases = [
        "从 Autoencoder 到 VAE：为什么普通 AE 不能自然生成",
        "概率建模直觉：$z$、$p(z)$、$q_\\phi(z\\mid x)$、$p_\\theta(x\\mid z)$ 分别是什么",
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

    bilingual_term_phrases = [
        "先看术语：英文保留，中文解释",
        "encoder（编码器）",
        "decoder（解码器）",
        "latent space（潜空间）",
        "posterior（后验分布）",
        "prior（先验分布）",
        "logits（未经过 sigmoid 的原始输出）",
        "forward（前向计算）",
        "backward（反向传播）",
        "optimizer（优化器）",
    ]
    for phrase in bilingual_term_phrases:
        assert phrase in joined_source

    visual_explanation_phrases = [
        "AE 和 VAE 的结构差异图",
        "重参数化技巧的计算路径图",
        "beta 对潜空间和采样的影响示意图",
        "这些图都是用 Matplotlib 在 notebook 里现场画出来的",
    ]
    for phrase in visual_explanation_phrases:
        assert phrase in joined_source

    code_comment_phrases = [
        "# 逐级向上寻找仓库根目录，保证从仓库根目录或 notebooks 目录启动都能导入源码",
        "# 这个小配置只用于离线学习和快速验证，不代表正式实验设置",
        "# 取出一个 batch（一小批样本），先检查数据长什么样",
        "# 编码器输出后验分布 q(z|x) 的两个参数：mu 和 logvar",
        "# 重参数化采样：把随机性放在 epsilon 里，z 仍然能对 mu/std 求梯度",
        "# 前向计算：图像 -> mu/logvar -> z -> recon_logits",
        "# 反向传播：根据 total loss 计算每个参数的梯度",
        "# 优化器更新：根据刚才算出的梯度修改模型参数",
    ]
    for phrase in code_comment_phrases:
        assert phrase in joined_source

    for line in joined_source.splitlines():
        if line.startswith("|"):
            assert count_unescaped_pipes(line) == 4, line

    assert "`beta=1` 时，`reconstruction + KL` 是负 ELBO" in joined_source
    assert "最小化它等价于最大化 ELBO" in joined_source
    assert "`beta != 1`（包括本作业的 `beta=0`）" in joined_source
    assert "beta-VAE 加权目标 / 消融目标" in joined_source


def test_educational_notebook_keeps_saved_learning_outputs():
    data = load_notebook(NOTEBOOK_PATH)
    code_cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]

    assert len(code_cells) >= 15
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert sum(1 for cell in code_cells if cell.get("outputs")) >= 8
    assert sum(1 for output in outputs if "image/png" in output.get("data", {})) >= 5
    assert any("text/plain" in output.get("data", {}) for output in outputs)


def test_educational_notebook_uses_latex_for_math_symbols():
    data = load_notebook(NOTEBOOK_PATH)
    markdown_source = "\n".join(
        "".join(cell.get("source", [])) for cell in data["cells"] if cell["cell_type"] == "markdown"
    )

    expected_latex_snippets = [
        r"$z$",
        r"$p(z)=\mathcal{N}(0,I)$",
        r"$q_\phi(z\mid x)$",
        r"$p_\theta(x\mid z)$",
        r"$\mu$",
        r"$\sigma$",
        r"$\log\sigma^2$",
        r"$\epsilon \sim \mathcal{N}(0,I)$",
        r"$z = \mu + \sigma \odot \epsilon$",
        r"$\mathrm{KL}(q_\phi(z\mid x)\,\|\,p(z))$",
        r"$-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$",
    ]
    for snippet in expected_latex_snippets:
        assert snippet in markdown_source

    assert "代码变量 `mu` 对应数学符号 $\\mu$" in markdown_source
    assert "代码变量 `logvar` 对应数学符号 $\\log\\sigma^2$" in markdown_source


def test_educational_notebook_matches_generator_source(tmp_path):
    generator_path = REPOSITORY_ROOT / "scripts/create_educational_notebook.py"
    spec = importlib.util.spec_from_file_location("educational_notebook_generator", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    generated_notebook = tmp_path / "01_vae_basics.ipynb"
    generator.write_notebook(generated_notebook)

    assert source_cells(load_notebook(generated_notebook)) == source_cells(load_notebook(NOTEBOOK_PATH))


def test_educational_notebook_generator_writes_clean_template(tmp_path):
    generator_path = REPOSITORY_ROOT / "scripts/create_educational_notebook.py"
    spec = importlib.util.spec_from_file_location("educational_notebook_generator", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    generated_notebook = tmp_path / "01_vae_basics.ipynb"
    generator.write_notebook(generated_notebook)
    data = load_notebook(generated_notebook)

    assert all(
        cell["execution_count"] is None and cell["outputs"] == []
        for cell in data["cells"]
        if cell["cell_type"] == "code"
    )


def test_educational_notebook_code_cells_execute_offline_on_cpu(tmp_path):
    workspace = tmp_path / "notebook-workspace"
    workspace.mkdir()
    (workspace / "src").symlink_to(REPOSITORY_ROOT / "src", target_is_directory=True)
    runner = tmp_path / "run_notebook.py"
    runner.write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "notebook_path = Path(sys.argv[1])\n"
        "workspace = Path(sys.argv[2])\n"
        "os.chdir(workspace)\n"
        "namespace = {'__name__': '__main__'}\n"
        "notebook = json.loads(notebook_path.read_text(encoding='utf-8'))\n"
        "executed = 0\n"
        "for index, cell in enumerate(notebook['cells']):\n"
        "    if cell['cell_type'] != 'code':\n"
        "        continue\n"
        "    source = ''.join(cell['source'])\n"
        "    exec(compile(source, f'<notebook cell {index}>', 'exec'), namespace)\n"
        "    import matplotlib.pyplot as plt\n"
        "    assert plt.get_fignums() == [], f'cell {index} left open Matplotlib figures'\n"
        "    executed += 1\n"
        "config = namespace['config']\n"
        "assert config['dataset'] == 'fake'\n"
        "assert config['device'] == 'cpu'\n"
        "assert config['download'] is False\n"
        "print(f'executed_code_cells={executed}')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")

    result = subprocess.run(
        [sys.executable, str(runner), str(NOTEBOOK_PATH), str(workspace)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "executed_code_cells=" in result.stdout


def test_notebook_import_cell_works_without_pythonpath(tmp_path):
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    import_cell = next(cell for cell in data["cells"] if cell["cell_type"] == "code")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")

    result = subprocess.run(
        [sys.executable, "-c", "".join(import_cell["source"])],
        cwd=NOTEBOOK_PATH.parent,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

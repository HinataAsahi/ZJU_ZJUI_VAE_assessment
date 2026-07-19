import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks/01_vae_basics.ipynb"


def test_educational_notebook_exists_and_has_required_sections():
    assert NOTEBOOK_PATH.exists()

    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
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

    assert "`beta=1` 时，`reconstruction + KL` 是负 ELBO" in joined_source
    assert "最小化它等价于最大化 ELBO" in joined_source
    assert "`beta != 1`（包括本作业的 `beta=0`）" in joined_source
    assert "beta-VAE 加权目标 / 消融目标" in joined_source
    assert all(
        cell["execution_count"] is None and cell["outputs"] == []
        for cell in data["cells"]
        if cell["cell_type"] == "code"
    )


def test_educational_notebook_matches_generator(tmp_path):
    generator_path = REPOSITORY_ROOT / "scripts/create_educational_notebook.py"
    spec = importlib.util.spec_from_file_location("educational_notebook_generator", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    generated_notebook = tmp_path / "01_vae_basics.ipynb"
    generator.write_notebook(generated_notebook)

    assert generated_notebook.read_bytes() == NOTEBOOK_PATH.read_bytes()


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

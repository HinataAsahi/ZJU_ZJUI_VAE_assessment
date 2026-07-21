# 提交检查清单

项目要求写的是通过邮件发送 zip 压缩包。最稳妥的提交方式是：邮件附件放干净的 zip 文件，同时在邮件正文里附上 GitHub 仓库链接。

不建议只发 GitHub 链接，除非老师或项目方明确回复允许这样提交。

## 推荐邮件正文

```text
Dear PIL Lab,

Please find attached my VAE assessment project zip package.

GitHub repository:
https://github.com/HinataAsahi/ZJU_ZJUI_VAE_assessment

The final report is included at:
reports/final_project_report.pdf

Best regards,
HinataAsahi
```

## 创建干净 zip

确认所有需要提交的文件都已经 commit 后，在仓库根目录运行：

```bash
git status --short
git archive --format=zip --output ../ZJU_ZJUI_VAE_assessment_submission.zip HEAD
```

`git archive` 只会打包当前 commit 中被 Git 跟踪的文件。它会排除本地数据集、checkpoint、`outputs/`、缓存、`.git/` 和个人未跟踪笔记。

## 发送前检查

发送邮件前运行：

```bash
PYTHONPATH=src pytest -v
pdfinfo reports/final_project_report.pdf
```

确认：

- 测试通过；
- `reports/final_project_report.pdf` 存在；
- README 包含训练和评估命令；
- 配置文件包含主要实验设置和随机种子；
- zip 文件可以正常打开。

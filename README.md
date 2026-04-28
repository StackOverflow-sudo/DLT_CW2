# COMP6252 CW2 - Planet Amazon Multi-label Classification

本项目用于 COMP6252 Coursework 2，选题来自 Kaggle 竞赛 **Planet: Understanding the Amazon from Space**。

任务类型是 **multi-label satellite image classification**：给定一张亚马逊地区的卫星图像，模型需要同时预测一个或多个标签，例如天气、植被、农业、水体、道路、居住地、采矿等。

当前项目已经包含：

- 数据读取与预处理代码
- 数据集可视化脚本
- baseline CNN 模型
- ImageNet-pretrained ResNet18 主模型
- 训练脚本
- 验证/测试脚本
- per-label 指标输出
- 报告素材目录

## 1. Project Structure

```text
.
|-- config/
|   |-- baseline_cnn.yaml
|   |-- cpu_quick.yaml
|   `-- default.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- README.md
|-- notebooks/
|-- outputs/
|-- reports/
|   |-- figures/
|   `-- report_outline.md
|-- scripts/
|   `-- download_data.ps1
|-- src/
|   `-- amazon_planet/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- evaluate.py
|       |-- explore.py
|       |-- metrics.py
|       |-- models.py
|       |-- train.py
|       `-- visualize_dataset.py
|-- .gitignore
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## 2. Data Preparation

本项目默认使用 Kaggle Planet 数据集中的 JPEG 训练图片和标签文件。

需要的数据文件是：

```text
train_v2.csv
train-jpg/
```

最终目录必须是：

```text
data/raw/
|-- train_v2.csv
`-- train-jpg/
    |-- train_0.jpg
    |-- train_1.jpg
    |-- train_2.jpg
    `-- ...
```

不要把 Kaggle 原始数据上传到 GitHub。`.gitignore` 已经忽略：

```text
data/raw/*
data/processed/*
outputs/*
others/*
*.pt
*.pth
*.ckpt
*.zip
*.7z
*.tar
*.torrent
```

如果已经配置 Kaggle CLI，可以尝试：

```powershell
.\scripts\download_data.ps1
```

但更简单的方法是从 Kaggle 网页手动下载：

```text
train_v2.csv.zip
train-jpg.tar.7z
```

解压后放到 `data/raw/`。

## 3. Environment Setup

建议在项目根目录执行所有命令：

```powershell
cd D:\DEVELOPMENT\courses\semester2\DeepLearningTechnology\CW2
```

安装依赖：

```powershell
pip install -r requirements.txt
```

安装本地项目包：

```powershell
pip install -e .
```

安装之后，可以直接运行：

```powershell
python -m amazon_planet.train --config config/default.yaml
```

如果没有运行 `pip install -e .`，则每次运行前需要临时设置：

```powershell
$env:PYTHONPATH='src'
```

## 4. GPU PyTorch Check

检查 PyTorch 是否能使用 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

如果输出类似：

```text
True
NVIDIA GeForce RTX 5060 Laptop GPU
```

说明 GPU 可用。

## 5. Configuration Files

### `config/default.yaml`

正式主模型配置：

```text
ImageNet-pretrained ResNet18
image size: 224
epochs: 5
batch size: 32
device: auto
```

输出：

```text
outputs/best_model.pt
outputs/metrics.json
```

### `config/baseline_cnn.yaml`

baseline 模型配置：

```text
Custom small CNN trained from scratch
image size: 128
epochs: 5
batch size: 64
device: auto
```

输出：

```text
outputs/baseline_cnn_best_model.pt
outputs/baseline_cnn_metrics.json
```

### `config/cpu_quick.yaml`

快速测试配置：

```text
ResNet18 from scratch
image size: 128
epochs: 1
device: cpu
```

用于检查训练流程是否能跑通。

## 6. Dataset Visualisation

生成报告用数据可视化：

```powershell
python -m amazon_planet.visualize_dataset --config config/default.yaml --output-dir reports/figures
```

输出文件包括：

```text
reports/figures/viz_label_frequency.png
reports/figures/viz_labels_per_image.png
reports/figures/viz_cooccurrence_all.png
reports/figures/viz_cooccurrence_weather.png
reports/figures/viz_cooccurrence_common_land.png
reports/figures/viz_cooccurrence_rare.png
reports/figures/viz_random_samples.png
reports/figures/viz_examples_by_label.png
reports/figures/viz_dataset_summary.md
reports/figures/viz_label_summary.csv
```

这些图适合放进报告的 data description / data visualisation 部分。

## 7. Train Models

### Train Baseline CNN

```powershell
python -m amazon_planet.train --config config/baseline_cnn.yaml
```

训练完成后会生成：

```text
outputs/baseline_cnn_best_model.pt
outputs/baseline_cnn_metrics.json
```

### Train Main ResNet18 Model

```powershell
python -m amazon_planet.train --config config/default.yaml
```

训练完成后会生成：

```text
outputs/best_model.pt
outputs/metrics.json
```

### Quick CPU Test

```powershell
python -m amazon_planet.train --config config/cpu_quick.yaml
```

这个命令只用于快速确认代码是否能跑，不建议作为正式实验结果。

## 8. Evaluate Models

### Evaluate Main Model

```powershell
python -m amazon_planet.evaluate --config config/default.yaml --checkpoint outputs/best_model.pt --per-label-output outputs/per_label_metrics.csv
```

输出包括：

- validation loss
- overall validation F2 score
- threshold
- average predicted labels per image
- best/worst label by F2
- per-label precision / recall / F1 / F2

per-label CSV 输出：

```text
outputs/per_label_metrics.csv
```

### Evaluate Baseline CNN

```powershell
python -m amazon_planet.evaluate --config config/baseline_cnn.yaml --checkpoint outputs/baseline_cnn_best_model.pt --per-label-output outputs/baseline_cnn_per_label_metrics.csv
```

输出：

```text
outputs/baseline_cnn_per_label_metrics.csv
```

## 9. Current Experiment Result

当前主模型是：

```text
ImageNet-pretrained ResNet18
```

已得到的 validation 结果：

```text
val_loss: 0.0903
threshold: 0.19
val_f2: 0.9238
validation samples: 8096
```

表现最好的标签包括：

```text
primary
clear
partly_cloudy
agriculture
```

表现较差的标签包括：

```text
slash_burn
blow_down
blooming
bare_ground
```

这说明整体 F2 较高，但稀有类别仍然困难，适合在报告中讨论 class imbalance。

## 10. Important Source Files

```text
src/amazon_planet/data.py
```

负责：

- 读取 `train_v2.csv`
- 读取 JPEG 图片
- 将标签转为 17 维 multi-hot vector
- 创建 train/validation DataLoader
- 图像 resize、normalization、augmentation

```text
src/amazon_planet/models.py
```

负责：

- `SimpleCNN` baseline
- ResNet18 / ResNet50 transfer learning model
- 替换最后一层为 17 标签输出

```text
src/amazon_planet/train.py
```

负责：

- 训练模型
- 计算 validation F2
- threshold tuning
- 保存最佳 checkpoint
- 保存 metrics JSON

```text
src/amazon_planet/evaluate.py
```

负责：

- 加载 checkpoint
- 在 validation set 上评估
- 输出整体指标
- 输出 per-label metrics

```text
src/amazon_planet/visualize_dataset.py
```

负责：

- 标签频率图
- 标签共现热图
- 标签数量分布图
- 随机图片样本图
- 每个标签示例图

## 11. GitHub Workflow

第一次初始化仓库：

```powershell
git init
git add .
git commit -m "Initial coursework project setup"
git branch -M main
git remote add origin git@github.com:StackOverflow-sudo/DLT_CW2.git
git push -u origin main
```

如果已经添加过 remote，检查：

```powershell
git remote -v
```

修改 remote：

```powershell
git remote set-url origin git@github.com:StackOverflow-sudo/DLT_CW2.git
```

日常提交：

```powershell
git status
git add .
git commit -m "Describe your change"
git push
```

创建新分支：

```powershell
git checkout -b feature/baseline-experiment
```

推送新分支：

```powershell
git push -u origin feature/baseline-experiment
```

拉取远程更新：

```powershell
git pull
```

## 12. Command List

### Setup

```powershell
cd D:\DEVELOPMENT\courses\semester2\DeepLearningTechnology\CW2
pip install -r requirements.txt
pip install -e .
```

### Check GPU

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Visualise Dataset

```powershell
python -m amazon_planet.visualize_dataset --config config/default.yaml --output-dir reports/figures
```

### Train Baseline

```powershell
python -m amazon_planet.train --config config/baseline_cnn.yaml
```

### Train Main Model

```powershell
python -m amazon_planet.train --config config/default.yaml
```

### Evaluate Baseline

```powershell
python -m amazon_planet.evaluate --config config/baseline_cnn.yaml --checkpoint outputs/baseline_cnn_best_model.pt --per-label-output outputs/baseline_cnn_per_label_metrics.csv
```

### Evaluate Main Model

```powershell
python -m amazon_planet.evaluate --config config/default.yaml --checkpoint outputs/best_model.pt --per-label-output outputs/per_label_metrics.csv
```

### Quick Test

```powershell
python -m amazon_planet.train --config config/cpu_quick.yaml
```

## 13. Suggested Report Structure

1. Introduction
2. Dataset and problem formulation
3. Data visualisation and reflection
4. Methodology
5. Baseline CNN
6. Transfer learning with ResNet18
7. Validation protocol and F2 score
8. Results and per-label analysis
9. Discussion of class imbalance and rare labels
10. Conclusion

Recommended experiment story:

```text
Dataset exploration
-> baseline CNN from scratch
-> ImageNet-pretrained ResNet18
-> threshold tuning
-> per-label analysis
-> discussion of rare labels and class imbalance
```

## 14. Notes for Team Members

- Do not commit Kaggle raw data.
- Do not commit model checkpoints.
- Do not commit large archives.
- Use branches for separate work.
- Pull before starting new work.
- Keep experiment results in `outputs/` locally.
- Put report-ready figures in `reports/figures/`.
- Update this README when adding new scripts or configs.

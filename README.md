# DiL:DCD + CACE for Semi-supervised Continual Learning

本项目提供离散对比蒸馏（DCD）与稀疏锚向量（CBSE/CACE）集成的半监督增量学习实现，支持 CIFAR-10/100、CUB-200、ImageNet-100，包含 DER 与非 DER 两条训练代码路径。

## 方法概要
- CBSE/CACE：生成组合块稀疏的锚向量（替代传统 ETF），为增量类别预留判别性特征空间。
- DCD：基于类别特征选择与离散化的对比蒸馏，对旧类强约束、新类弱约束，缓解遗忘、提升无标数据利用。
- 无标数据与重放：配合伪标签、缓冲区重放与一致性训练，兼顾稳定性与可塑性。

## 项目结构
- `train_semi.py` / `train_semi_der.py`：非 DER / DER 主训练入口。
- `train_cub.py`：CUB-200 训练入口。
- `utils_incremental/discrete_contrastive_distillation.py`：DCD 核心模块。
- `utils_incremental/cbse.py`：CBSE 锚向量生成。
- `utils_incremental/incremental_train_and_eval_semi.py`：增量训练与评估逻辑。
- `utils_pytorch.py`：通用工具（含 CBSE 集成）。
- 运行脚本：`run_cifar.sh`, `run_cub.sh`, `run_imagenet.sh`。
- 文档：`DCD_README.md`, `DCD_IMPLEMENTATION_SUMMARY.md`, `DCD_CIFAR10_vs_CIFAR100_ANALYSIS.md` 等。

## 环境依赖
- Python 3.10+
- PyTorch 2.1.x, torchvision 0.16.x
- 其余依赖见 `environment.yml`
- 推荐使用 Conda：`conda env create -f environment.yml` 并激活对应环境。

## 数据准备
- CIFAR-10/100：使用 torchvision 内置下载或本地缓存。
- CUB-200：按官方目录结构放置 train/val。
- ImageNet-100：按官方 split 组织；确保 `--data_dir` 指向解压路径。
- 确认 `--image_size`（32 或 224）与数据集匹配。

## 关键参数（常用）
- 数据与模型：`--dataset {cifar10,cifar100,cub,imagenet100}`, `--model {resnet18,resnet20,resnet32}`, `--image_size`
- 增量设置：`--nb_cl_fg`（首任务类数），`--nb_cl`（每阶段新增类数），`--num_classes`
- 训练轮次：`--epochs`（首任务），`--epochs_new`（增量）
- 重放与无标：`--buffer_size`, `--include_unlabel`, `--p_cutoff`, `--u_iter`, `--use_ulb_kd`, `--ulb_kd_mode similarity`
- DCD：`--enable_dcd`, `--lambda_dcd`, （可选）`--dcd_top_k_class`, `--dcd_top_k_sample`, `--dcd_alpha`, `--dcd_temperature`
- 其他：`--warmup_epochs`, `--random_seed`, `--train_batch_size`, `--test_batch_size`, 学习率与调度

## 训练与评估提示
- 训练脚本与示例参数可参考 `run_cifar.sh`, `run_cub.sh`, `run_imagenet.sh`（已包含 DCD/CBSE 设置）。
- DER 与非 DER 分别使用 `train_semi_der.py` 与 `train_semi.py`。
- 评估/续训：在对应命令中添加 `--resume --model_path <checkpoint>`，并将 `--epochs`/`--epochs_new` 设为 0 进行纯评估。
- 日志与模型：默认输出到 `log/` 与 `checkpoint/`（目录不存在会自动创建）。


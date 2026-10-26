# RTX 4090 训练交接

代码分支：`codex/pretrained-auroc-improve`。后续正式训练由队友在 4090 上执行；本次开发在服务器独立目录验证。实际服务器测试范围和日志见 [验证记录](SERVER_HANDOFF_VALIDATION.md)。服务器 GPU 验证使用 V100，4090 尚需队友实测，跨硬件不保证逐位一致。

已训练结果、原始日志与历史权重获取方法见 [正式结果报告](TRAINING_RESULTS.md) 和 [材料目录](../training_records/20261022-23/README.md)。可以下载已有B0固定best作为新refine的`--init-from`起点；已有refine best可供推理验证。旧实验尚存的材料与缺口分别保留，不要求为了材料整理而重训。

新开发按 [规则核查](COMPLIANCE_REVIEW.md) 和 [后续计划](DEVELOPMENT_PLAN.md) 执行；最终T4 16GB上的100ms端到端约束与4090训练速度分别验证。下文仍是现有已实现配方的命令，计划中的512/DINO配置不能当作已可用命令。

## 取得代码与环境

在队友自己的机器执行，以下命令兼容 Bash/PowerShell（每条为一行）：

```text
git clone --branch codex/pretrained-auroc-improve https://github.com/InkyUmbrella/2026Ncccu_BiDataCompetition.git
cd 2026Ncccu_BiDataCompetition
conda create -n glaucoma4090 python=3.11 pip -y
conda activate glaucoma4090
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-server-tested.txt
python -m pip install -e . --no-deps --no-build-isolation
python -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

已有仓库先 `git fetch origin`，再切换此分支并 `git pull --ff-only`。保留个人未提交修改。CUDA wheel 安装组合来自 [PyTorch 官方 2.5.1 安装说明](https://pytorch.org/get-started/previous-versions/#v251)。需要兼容 CUDA 12.1 的 NVIDIA 驱动；无需另外安装系统 CUDA Toolkit。已有可用环境先比对版本，避免覆盖共享环境。

`requirements-server-tested.txt` 固定服务器已验证的直接依赖；完整实际安装版本会随每次训练写入 `metadata.json`。新机器安装结果应以该文件记录为准。

在队友**自己的单卡 4090 机器**选择 GPU 0：

```bash
# Bash
export CUDA_VISIBLE_DEVICES=0
```

```powershell
# PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"
```

共享服务器目前只允许本实验使用物理 GPU 4/5，必须先确认目标卡空闲。上述 GPU 0 示例仅适用于队友自己的机器。

## 数据和初始化

将官方数据放在 `dataset/glaucoma_first/`，其下为 `train.csv`、`valid.csv` 及对应图片目录。默认训练只读取 train/valid；保持官方 51,396/5,849 张及 191 个验证阳性，不改变标签和划分。各命令的 `--data-root` 可换为自己的绝对路径（带空格时加引号）。

第一次 B0 从官方 `IMAGENET1K_V1` 初始化，联网时 torchvision 自动下载到 Torch 缓存。离线机器可先拷贝官方缓存文件 `convnext_tiny-983f1562.pth` 到 `torch.hub.get_dir()/checkpoints/`，查询缓存根目录：

```text
python -c "import torch; print(torch.hub.get_dir())"
```

本项目使用的官方文件 SHA256：

```text
983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
```

每次运行都会保存实际完整模型初始状态；官方预训练文件、`--init-from` 或 `--resume` 输入也会按原字节复制归档并计算 SHA256。新随机分类头的初始状态因此也有证据，不只记录预训练骨干名称。

可选：先在新机器的本地 SSD 建 384 无损缓存（约 24 GiB），保持随机增强在线执行：

```text
python -m glaucoma cache-images --config configs/experiments/b0_crop384_imagenet.json --data-root dataset/glaucoma_first --output outputs/cache_crop384 --workers 4
```

随后给训练命令追加 `--image-cache outputs/cache_crop384`。缓存绑定源路径、文件大小/mtime 和 CSV 哈希，迁移数据后应重新建立；不要直接套用另一台机器的缓存 manifest。默认仍完整核查图片，首次审计阶段可能耗时较长。

## 正式启动顺序

所有命令在仓库根目录运行。以下启动器在前台运行并自动保存完整控制台日志、实际参数、子进程 PID 和退出码。长训练请放在不会中断的终端或 tmux 会话中。

第一阶段：B0，官方 ImageNet 初始化，30 轮：

```text
python scripts/train_audited.py --config configs/experiments/b0_crop384_imagenet.json --data-root dataset/glaucoma_first --output-root outputs/b0_4090 --seed 42
```

输出第一行 `[launch]` 是本次启动记录目录；日志中的 `[run]` 是实际模型运行目录，启动记录 `launch.json` 也会保存 `run_directory`。等退出码为 0 后，将下方 `B0_RUN` 换成该目录。

第二阶段：从**此次 B0 的 best.pt**开始独立 10 轮低学习率微调，同时评估 raw/EMA：

```text
python scripts/train_audited.py --config configs/experiments/b0_crop384_refine_ema.json --data-root dataset/glaucoma_first --output-root outputs/refine_4090 --init-from B0_RUN/best.pt --seed 42
```

| 配方 | 轮数 | 分类头最高 LR | 骨干最高 LR | EMA |
|---|---:|---:|---:|---:|
| B0 | 30 | 1e-4 | 1e-4 | 关闭 |
| refine | 10 | 1e-5 | 1e-5 | 0.999 |
| restart（对照） | 15 | 1e-4 | 2e-5 | 0.999 |

均为 ConvNeXt-Tiny 纯图像、384 裁黑边/等比补边、ImageNet 归一化、seed42、microbatch32 × 累积4（有效 batch128）、AdamW、weight_decay0.05、pos_weight5、1轮 warmup/余弦衰减。先保留这些设置，不因换卡任意加大有效 batch。

历史服务器结果：B0 AUROC **0.986505**；refine **0.987328**（第1轮 raw），相对 B0 的配对 bootstrap 95% 区间跨 0；restart **0.983192**（第6轮 raw）。这些是旧验证结果，不能作为队友新一次训练的结果。当前顺序优先 B0→refine，restart 配置保留作可选对照：

```text
python scripts/train_audited.py --config configs/experiments/b0_crop384_imagenet_backbone_ema.json --data-root dataset/glaucoma_first --output-root outputs/restart_4090 --seed 42
```

每轮用相同完整 valid 分别评估 raw/EMA，按 AUROC、并列时 PR 梯形面积选模；`best.pt` 是选中的普通单模型，不需要 EMA 包装推理。AP 不等同于 PR 梯形面积，最高 AP 轮次也不等同于 AUROC 最佳轮次。

## 中断恢复与完成检查

恢复必须使用**相同配方和参数**对应的完整断点，例如 B0：

```text
python scripts/train_audited.py --config configs/experiments/b0_crop384_imagenet.json --data-root dataset/glaucoma_first --output-root outputs/b0_4090 --resume PREVIOUS_RUN/last.pt --seed 42
```

refine 恢复时改用 refine 配置和它自己的 `last.pt`。不要同时传 `--init-from`。恢复包含模型、优化器、GradScaler、Python/NumPy/Torch/CUDA RNG、历史、内嵌最佳权重，以及启用时的 EMA state/n_averaged；学习率计划保持原定总轮数。AMP 跳过的更新不推进 EMA。

每次恢复创建新目录，checkpoint 的 epoch/history 延续原进度。**保留所有父运行目录和对应 launches 目录**，以合并形成从第1轮开始的完整证据链。新目录只记录此次实际执行的轮次，`initialization.json` 指向父运行并归档输入断点。若写入 `last.pt` 时异常，确认已完整保存的最新 `checkpoints/epoch_NNNN.pt` 也可作为 `--resume` 输入；不能用 `best.pt` 代替完整恢复断点。数据路径/输出路径允许迁移，其他配置需一致，CSV 内容必须一致。

首次小试可在原命令追加 `--stop-after-epochs 1`，它只限制本次运行轮数，不缩短总学习率计划。接着用其 `last.pt` 恢复即可完成原计划；该小试仍属正式数据训练的一部分，应保留全部证据。

检查产物哈希（纯 CPU，不占训练 GPU）：

```text
python -m glaucoma verify-run --run-dir RUN_DIRECTORY
```

该命令逐文件重算 SHA256，发现缺失/被改动文件就非零退出。只有 `training_complete=true`、`completed_epochs` 达到30/10/15、启动器 `train.exit=0`，且验证通过，才能称该配方完成。每次中断父目录也分别核验；`verify-run` 的 `recorded_epochs` 仅指本次目录。

## 必须保留的交接材料

```text
OUTPUT_ROOT/launches/TIME_ID/
  launch.json                  # 完整命令、GPU映射、实际run目录、时间、退出码、日志哈希
  train.log                    # stdout+stderr全日志，实时flush
  train.exit
RUN_DIRECTORY/
  config.json / metadata.json  # 实际配置、seed、环境版本、CSV哈希、Git及源码信息
  source.tar.gz / source_manifest.json
  initialization.json         # 实际初始化来源、SHA256与父运行
  initialization/model_initial.pt
  initialization/source.pt 或 source.pth  # 有外部初始化输入时
  checkpoints/epoch_0001.pt ...           # 每轮完整恢复断点，拒绝覆盖
  checkpoint_manifest.json    # 从1开始epoch、UTC保存时间、mtime_ns、字节数、SHA256
  best.pt / last.pt
  data_audit.json / age_stats.json / history.json
  metrics.json / valid_predictions.csv
  artifact_manifest.json      # 完成/部分完成时的产物哈希清单
```

`history.json`/`metrics.json`/checkpoint 内的 epoch 沿用代码的从0开始约定；`checkpoint_manifest.json` 的 epoch 和文件名从1开始。`verify-run` 负责文件完整性校验，不验证成绩真实性或比赛平台提交关系；时间戳来自运行主机，不是第三方时间认证。

每轮保存完整训练状态，会额外占用磁盘。粗略预留 B0 30轮约13–15 GiB，refine 10轮约6–7 GiB，restart 15轮约9–10 GiB，另加初始化、last/best、数据和缓存；以实际文件大小为准。不要自动删除较早轮次。Git 仅保存源码/配置/文档，训练产物另行备份。

这些归档能力用于**未来的新运行**；旧实验没有逐轮留存的文件无法事后补造。用户目前尚未提交比赛预测。本次交接不生成或提交比赛预测，也不自动启动上述完整训练。

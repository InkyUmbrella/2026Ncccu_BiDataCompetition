# ConvNeXt-Tiny 预训练权重微调

本实验加载 torchvision 官方的 `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`，再使用比赛 train 的青光眼标签微调。它使用已经在 ImageNet-1K 上训练好的图像特征，不是从零训练一个新的预训练模型。ImageNet 的 1000 类分类层被丢弃，现有二分类头重新初始化；骨干和分类头都参与微调。

## 权重来源与实验边界

- [torchvision 官方模型及权重说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)：固定使用 `IMAGENET1K_V1`，不用可能随版本变化的 `DEFAULT`。
- [官方权重文件](https://download.pytorch.org/models/convnext_tiny-983f1562.pth)：约 109 MiB，首次训练由 torchvision 下载并校验文件名中的 SHA256 前缀，之后复用 torch hub 缓存。
- [比赛规则](https://www.ncccu.org.cn/index/Paper/case2.html)（2026-10-22 核对）：只允许主办方训练数据，禁止外部标注数据，但没有明确公开预训练权重的适用范围。这是外部预训练实验；是否能用于参赛提交需由主办方明确。
- 训练只使用 train；valid 用于选模、评估；test 不参与训练和预处理统计量拟合。实验不能保证 AUROC 达到 0.98。

## 配置与对照

| 配置 | 初始化 | 用途 |
|---|---|---|
| `configs/experiments/b0_crop384_imagenet.json` | ImageNet-1K V1 | 第一次预训练微调 |
| `configs/experiments/b0_crop384_random_control.json` | 随机 | 同配方对照，隔离初始化效果 |

两份配置只在实验名和权重初始化上不同：纯图像、去黑边后等比缩放补边至 384、ImageNet 归一化、原有温和增强、seed=42、30 epochs、batch=32、梯度累积=4（有效 batch=128）、AdamW、学习率 1e-4、1 epoch warmup、余弦衰减、weight decay=0.05、`pos_weight=5`、V100 FP16。它们是第一轮对照起点，不代表最优超参数；随机模型可能需要更多训练及不同学习率。

归一化采用 `mean=[0.485, 0.456, 0.406]`、`std=[0.229, 0.224, 0.225]`。保留眼底完整视野，不套用 ImageNet 分类的中心裁剪。归一化只发生在读取 RGB 图片/缓存并完成增强之后，已有同尺寸、同裁边配方的无损缓存可复用。

旧配置没有 `model.weights` 和 `data.normalization` 时，默认仍是随机初始化和原来的 `mean=std=0.5`；旧 checkpoint 使用原预处理。新 checkpoint 保存权重标识和归一化方式，评估、预测与恢复训练直接使用其中的模型参数，不重新下载 ImageNet 权重。`--init-from` 也以指定 checkpoint 为准，并拒绝归一化不一致的迁移。

## 服务器运行

改动基于 `fwq` 的 `9141f72`，服务器实验目录为 `/home/lichunyu/competition-pretrained`，正式 `/home/lichunyu/competition` 不切换分支。使用原有 `glaucoma` 环境，不升级 CUDA/PyTorch。下例以空闲 GPU 2 为例；启动前用 `nvidia-smi` 检查当前占用。

```bash
cd /home/lichunyu/competition-pretrained
conda activate glaucoma
export PYTHONPATH=src

# 可先单独下载并校验，训练入口也会自动执行这一步。
python -c "from torchvision.models import ConvNeXt_Tiny_Weights; w=ConvNeXt_Tiny_Weights.IMAGENET1K_V1; w.get_state_dict(progress=True, check_hash=True); print(w.url)"

# 新实验不使用旧的随机初始化模型 checkpoint。
CUDA_VISIBLE_DEVICES=2 python -m glaucoma train \
  --config configs/experiments/b0_crop384_imagenet.json \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first
```

需要先检查一个完整 epoch 时追加 `--stop-after-epochs 1`。这仍会读取完整训练集和验证集，不能当作一个 batch 的 smoke test。随后用同一配置加 `--resume /实际运行目录/last.pt` 继续，保留原定 30 epochs 的学习率计划。

需要后台运行时：

```bash
mkdir -p outputs/pretrained
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
  /home/lichunyu/miniconda3/envs/glaucoma/bin/python -u -m glaucoma train \
  --config configs/experiments/b0_crop384_imagenet.json \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  > outputs/pretrained/train_imagenet.log 2>&1 < /dev/null &
echo $!
tail -f outputs/pretrained/train_imagenet.log
```

保存日志时请使用新的文件名，避免覆盖已有实验日志。模型选择继续使用验证 AUROC，同分再看 PR 梯形面积；`ap` 和 `auprc_trapz` 分别保存。结果位于日志 `[run]` 指定的目录：`history.json` 查看逐轮变化，`metrics.json` 查看最佳验证结果，`best.pt` 用于评估/预测，`last.pt` 用于恢复训练。

对照实验将配置替换为 `b0_crop384_random_control.json`，其他参数保持一致。先观察单次实验是否有明显提升，再对有希望的方案增加随机种子；不要仅凭一个验证集上的微小差值断言有效。

## 验证

离线回归测试不下载外部权重，覆盖权重实际注入编码器、官方归一化、缓存一致性，以及预训练实验的离线恢复/迁移/评估/预测：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=src python -c \
  "import torch, unittest; torch.set_num_threads(2); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

真实官方下载、V100 FP16 和实际图片的 smoke test 应另行记录。这些工程检查不能代替完整训练的验证 AUROC/AP，也不能证明 T4 的端到端延迟满足比赛要求。

## 2026-10-22 执行记录

- 官方文件已在服务器下载，114,419,221 bytes；完整 SHA256 为 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`。
- 服务器原环境 PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121：回归测试 23 项通过，原有 opt-in GPU 测试 1 项跳过；预训练 GPU 路径单独实测。
- V100 GPU 2：180 个编码器参数/缓冲张量与官方权重逐一完全相同。128 张真实训练图（4 阳性）、32×4 累积、FP16 完成 1 次优化器更新，编码器权重确实改变，参数全部有限，优化器跳步为 0。峰值已分配张量显存 5,885.1 MiB。另完成 32 张真实验证图片的前向；不将该小样本当作正式 AUROC/AP。
- 已启动“建立缓存 → 30 epoch 微调”后台流程，启动时 launcher PID 为 `2212590`。缓存为 `/var/tmp/glaucoma-pretrained-lichunyu-crop384-20261022`，train/valid 共约 24 GiB；它保存裁边缩放后的 RGB uint8 图片，每轮随机增强仍重新执行。
- 缓存成功后会检查 GPU 2 是否仍空闲；若已被占用，记录原因并退出，不启动训练。正式原目录及 GPU 0、1 的原有训练不变。

执行状态以服务器实时文件为准，文档不代表后台任务已完成：

```bash
tail -n 30 /home/lichunyu/competition-pretrained/outputs/pretrained/train_imagenet.log
cat /home/lichunyu/competition-pretrained/outputs/pretrained-validation/launch.json
# 流程退出时才会生成此文件，0 表示成功；不存在时结合进程/日志确认状态。
cat /home/lichunyu/competition-pretrained/outputs/pretrained-validation/train.exit
```

校验报告在服务器 `outputs/pretrained-validation/`：`official_weights.json`、`cpu-tests.log`、`cuda-real-images.json`。本地 `outputs/pretrained-validation/` 保存报告副本。后台启动脚本及其日志也保存在该目录对应的输出路径中，不提交权重、图片或缓存到 Git。

## 设备解析修复与继续训练

首次任务完成全部缓存后，在正式训练入口的 `resolve_device()` 中退出：`torch.cuda.set_device()` 不接受没有明确编号的 `torch.device('cuda')`。原 smoke test 使用了显式 `cuda:0`，因此没有覆盖这个正式入口问题。

现已修复为：对无编号的 CUDA 设备，先用 `torch.cuda.current_device()` 得到当前逻辑编号，再调用 `set_device()`；显式卡号和 CPU 处理保持原语义。当 `CUDA_VISIBLE_DEVICES=2` 时，物理 GPU 2 对应进程内的逻辑 `cuda:0`。

新增 `tests/test_device.py` 覆盖默认 CUDA、显式索引、CPU 和 CUDA 不可用四种路径。修复前默认配置测试可稳定复现失败；修复后服务器回归测试 27 项通过，原 opt-in GPU 测试 1 项跳过，且真实 V100 上默认配置已成功解析为 `cuda:0`。

原失败发生在第 1 个 epoch 之前；本次使用已下载的 ImageNet 权重及已完成的 57,245 张缓存，从第 1 轮开始训练，计划共 30 轮。最新运行目录为：

```text
/home/lichunyu/competition-pretrained/outputs/pretrained/b0_crop384_imagenet/seed42_20261022T103749Z_393b6b2d
```

最新 launcher PID 为 `2223490`，训练子进程启动时 PID 为 `2223494`。原失败日志、退出码和启动记录保存在 `outputs/pretrained-validation/failure_before_device_fix_20261022T103742Z/`；当前记录仍使用 `outputs/pretrained-validation/launch.json`、`train.exit` 及 `outputs/pretrained/train_imagenet.log`。日志路径不变，可继续使用上面的 `tail` 命令。

新启动脚本 `outputs/pretrained-validation/continue_imagenet.sh` 直接复用缓存，先检查 GPU 2 是否空闲，再调用正式训练入口。修复验证报告为 `outputs/pretrained-validation/device-fix-tests.log`。当前运行的指标以该运行目录内的 `history.json` 和 `metrics.json` 为准；不要把离线合成测试的指标当作正式成绩。

续训验收：train 51,396 张、valid 5,849 张缓存全部通过审计，错误数为 0。正式训练入口已推进到第 1 轮 `batch=100/1607`，当时累计训练损失为 `0.708956`，物理 GPU 2 利用率约 99%、显存占用 6,707 MiB。这证明正式入口已越过原报错并开始训练；完整验证 AUROC/AP 需等待该轮训练和验证结束。每 10 分钟的自动跟踪已恢复。

# fwq 模型与预训练权重的集成验证

集成分支为 `codex/fwq-pretrained-integration`，基于 GitHub `fwq` 的 `9141f72f0e8d8382e87c1d9d813bdfdfd9dabe26`。
保留 fwq 的 B0、年龄拼接和年龄残差模型，合入 ImageNet 初始化、归一化、checkpoint 加载及默认 CUDA 设备解析支持。
原有配置继续使用原来的随机初始化与对称归一化。预训练实现与工程验证见 [PRETRAINED.md](PRETRAINED.md)。

## 当前实验安排

1. 原 `/home/lichunyu/competition-pretrained` 中的 B0 继续完成 30 轮。
2. 将当前可完整读取的 `best.pt` 固定为独立快照，记录 SHA256、原运行目录、最佳 epoch、指标、配置和代码版本。后续源目录更新最佳权重不会改变这个快照。
3. 在单独的 `/home/lichunyu/competition-fwq-pretrained` 中，用快照评估合并后的 B0，读取全部 5,849 张 valid 图片。
4. B0 验证通过后，从同一快照初始化年龄残差模型，在另一张确实空闲的 GPU 上训练 20 轮；它与原 B0 的剩余训练并行。使用 fwq 的学习率、有效 batch=128、前 3 轮冻结图像分支和图像学习率倍率 0.1。
5. 年龄残差训练后，使用其最佳权重重新评估完整 valid。原 B0 的 30 轮完成后，再评估最终 B0 最佳权重；若与快照完全相同则复用已完成的 B0 评估。

GPU 空闲必须同时核对计算进程和显存占用。利用率为 0 或只有少量显存占用的卡，仍可能属于其他任务。
2026-10-22 检查时，GPU 2 正在运行原 B0，GPU 5 没有计算进程；其他卡已经有进程。启动时仍需重新检查。
图像缓存可复用 `/var/tmp/glaucoma-pretrained-lichunyu-crop384-20261022`，并持续观察磁盘 I/O 和真实进度。

## 预处理与迁移语义

B0 的完整骨干和二分类头可严格加载；年龄残差继承相同的图像参数，新增年龄分支的末层为零初始化，随后实际训练。
年龄拼接模型的分类头输入维度不同，不属于本轮两个实验。

所有迁移实验保持 RGB、去黑边（阈值 10、边距 0.03）、等比缩放补边至 384，以及 ImageNet mean/std。
`best.pt` 中的模型配置和归一化由推理入口恢复，已有权重的评估、恢复和迁移不重新下载 ImageNet 文件。
训练和年龄统计只使用 train；选模和对比使用官方 valid；本轮不使用 test 生成提交。

## 命令

以下在合并后的服务器目录中运行，`B0_SNAPSHOT` 为已固定并核验的快照路径，GPU 5 仅为启动时已确认空闲的示例。
每个评估目录必须是新路径；后台日志、PID、退出码和运行目录记录在 `outputs/fwq-transfer/`。

```bash
cd /home/lichunyu/competition-fwq-pretrained
export PYTHONPATH="$PWD/src"

CUDA_VISIBLE_DEVICES=5 python -m glaucoma evaluate \
  --checkpoint "$B0_SNAPSHOT" \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --device cuda:0 --batch-size 32 --num-workers 4 --split valid \
  --output-dir outputs/fwq-transfer/b0_snapshot_evaluation_RUN

CUDA_VISIBLE_DEVICES=5 python -m glaucoma train \
  --config configs/experiments/b0_crop384_age_residual_imagenet.json \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --image-cache /var/tmp/glaucoma-pretrained-lichunyu-crop384-20261022 \
  --init-from "$B0_SNAPSHOT"

CUDA_VISIBLE_DEVICES=5 python -m glaucoma evaluate \
  --checkpoint "$AGE_RUN/best.pt" \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --device cuda:0 --batch-size 32 --num-workers 4 --split valid \
  --output-dir outputs/fwq-transfer/age_residual_evaluation_RUN
```

新增配置为 `configs/experiments/b0_crop384_age_residual_imagenet.json`。它保留 fwq 的年龄残差训练配方，并指定本次迁移的 ImageNet 归一化与独立输出目录。
年龄残差阶段的 `--init-from` 使用固定 B0 快照，恢复年龄实验则使用该实验的完整 `last.pt`。

## 验收与对比

- B0 评估必须有 5,849 行、191 个阳性；预测名称与 CSV 顺序一致，概率及 AUROC/AP/PR 梯形面积均有限。
- 将快照 B0 的新评估与快照保存的指标比较。若差异明显，先排查数据、预处理、模型参数和 FP16/批次行为，再启动年龄训练。
- 年龄实验必须实际完成 20 轮、正常退出，并保存可加载的 `best.pt`、完整 `last.pt`、历史、验证预测和指标。冻结与解冻行为由回归测试覆盖。
- 对比快照 B0、最终 B0、训练后的年龄残差，分别列出 AUROC、AP 和 `auprc_trapz`。年龄实验的初始化快照须明确列出；若最终 B0 权重发生变化，两个结果不作为相同起点的严格消融实验。
- 合成数据测试和初始化时相同的输出只验证工程兼容性，正式成绩来自完整 valid。年龄分支效果较差时保留结果及 B0 最佳权重，不自动展开其他模型搜索。

只有原 B0 30 轮及本轮验证、年龄残差训练和评估全部完成后，才结束自动跟踪。

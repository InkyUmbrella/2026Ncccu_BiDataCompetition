# 512 → 640 分辨率对照（新机器运行）

这轮不增加注意力或年龄分支。round3 两个分支均未超过初始模型，最终都回退到初始权重。当前已独立复核的最佳单模型 AUROC 为 0.9902695344959368；固定双模型等权 logits 融合为 0.9911860887331841，尚未证明稳定收益。

本轮两个实验仅 image_size 和实验名称/输出路径不同：512 对照、640 实验。均从同一份 best.pt 初始化，10 epochs，head LR 1e-5、backbone LR 1e-6，warmup 1 epoch，EMA .999 + raw 比较，pos_weight=5，自然采样。默认 micro batch8 × accumulation16 = effective batch128。降低骨干学习率是本轮两组共有设置，因此比较的是本轮512与640，而不是直接与上轮普通微调比较。

640 提供更细图像信息，但不保证提高分数；也会提高推理开销。未测试 T4 100ms 限制。

## 1. 准备代码、数据和权重

```sh
git clone --branch shimmer https://github.com/InkyUmbrella/2026Ncccu_BiDataCompetition.git
cd 2026Ncccu_BiDataCompetition
python -m pip install -e .
```

已有仓库则切换到 shimmer 后 `git pull --ff-only`。安装 PyTorch 时按新机器驱动选择对应版本；已在 torch2.5.1 / torchvision0.20.1 验证主要流程。

**best.pt 不在 GitHub，必须从旧机器迁移。** 使用原始纯图像 seed42 最佳权重，而非 round3 的残差模型包装：

```text
/var/tmp/heyuhang-shimmer-round2-20261028/jobs/seed42/training/shimmer_512_pretrained_reference/seed42_20261027T191640Z_bd3a1b70/best.pt
SHA256: b546ffa77040c879ea6ec0b25dc27cd62e402e9c07e7efeaca3bd494de3f119e
大小: 111732674 bytes
```

数据根目录需包含 train.csv、valid.csv 和对应图片目录，沿用原比赛数据。程序记录权重及CSV哈希。初始化、推理均不会重新下载 ImageNet 权重。

## 2. 一条命令顺序运行两组

替换以下三个路径，GPU编号由运行者指定：

```sh
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_resolution_ablation.py \
  --checkpoint /path/to/best.pt \
  --data-root /path/to/glaucoma_first \
  --output-root /path/to/outputs/resolution_seed42 \
  --cache
```

脚本在选定的一张卡上依次执行：复核原始512 → 512对照训练与独立重评 → 640训练与独立重评。不会自动占其他卡。output-root 必须是不存在的新目录。

`--cache` 会在输出目录内为每个分辨率构建无损PNG缓存，建议输出目录放本地SSD。两个分辨率合计建议至少预留80GB磁盘（实际取决于图片压缩率）；脚本不自动放入 /dev/shm，避免占满内存。使用本地原始图片时可省略 --cache；速度取决于解码与磁盘。

显存不足时，两组一起改为 `--batch-size 4`，脚本自动改为 accumulation32，仍保持有效batch128。只能选128的正整数因数。不做自动OOM重试，以免改变单组训练设置而破坏对照。可用 --workers 调整数据加载进程，默认4。

若有两张可用卡，可各用一个新输出目录，分别指定 `--sizes 512` / `--sizes 640` 并行跑；起始权重、batch-size及workers应一致。

只检查权重、数据表并生成配置，不使用GPU训练：

```sh
python scripts/run_resolution_ablation.py \
  --checkpoint /path/to/best.pt --data-root /path/to/glaucoma_first \
  --output-root /path/to/outputs/resolution_preflight --prepare-only
```

正式运行另用一个新 output-root。

## 3. 查看与判断结果

```sh
tail -f /path/to/outputs/resolution_seed42/512/train.log
```

640日志位于640/train.log；独立重评日志为各目录evaluate.log。结果汇总在根目录 manifest.json，包含：原始512复核分数、每组训练前分数、每组最佳权重的独立重评分数、相对原始512的AUROC差值、权重哈希。

640会在训练前对初始权重重新评估，保留这个640起点作为候选。**这个保护不会保证640优于原始512**，所以最终也必须比较baseline512。selected_weights=initial 表示训练没有超过该分辨率的初始模型，不能作为训练成功的证据。

程序核对完整验证集的图像名、顺序、标签和训练CSV哈希，检查独立重评AUROC/AP/AUPRC与训练保存结果一致。重复运行不同种子才可判断训练稳定性；本轮结果不能直接证明测试集提升。原始权重文件不会被覆盖。

中断后可用现有CLI的 `train --config <该组config.json> --resume <该组last.pt>` 继续。不要同时传 --init-from；resume保持原有数值配置。此方式完成后需另用CLI evaluate独立重评，启动脚本不会自动接管手动恢复的任务。

回传两组 manifest.json、config/metadata/history/metrics/initial_validation/独立重评metrics 即可；比赛图片、逐样本预测和模型权重不要提交Git。

## 4. 本轮实测结果（Windows 单卡，seed42）

两组均从同一 best.pt 初始化、10 epochs 跑完，权重与预测未提交 Git。数值均为**同一台机器、同一 evaluate 协议**下产出，可横向比较。

| 方案 | best AUROC | best AP | 相对初始权重 |
|---|---|---|---|
| 初始 512 权重（本机复核 baseline512） | 0.988501 | 0.827547 | — |
| 本轮 512 微调（best epoch 4） | 0.988612 | 0.829911 | +0.0001 |
| 本轮 640 微调（独立重评） | 0.982846 | 0.811048 | −0.0057 |

- 512 微调全程 0.9885–0.9886，与初始权重等价；640 从 0.9812 爬到 0.9828 后封顶，两者差距稳定在 0.006 附近，曲线不交叉。
- 结论：在本轮“head 1e-5 / backbone 1e-6、10 epochs”的受控设置下，**640 没有收益且明显更差**，应继续沿用初始 512 权重。可能原因是 640 输入与 ImageNet 预训练尺度偏离更大，而骨干学习率过低、epoch 太少，模型没有余量适配新尺度。若要再试，需提高 backbone LR（5e-6 ~ 1e-5）单独成轮，并补 T4 延迟测试。

**口径提醒**：本文开头记录的 0.99027 来自旧机器评估协议，本机对同一 best.pt 复核只得 0.988501。跨机器数字不可混用，否则会把 512 微调误判为退步。后续结论一律以同机同协议的三行数字为准。

### Windows 上的 dataloader 死锁（已定位）

本轮出现 4 次训练卡死，全部发生在 **epoch 边界重建 dataloader 时**：`num_workers>0` 的 spawn 子进程在 `import torch` 阶段崩溃（`640_train.log.err` 中可见 `SystemError: error return without exception set`），父进程无限等待该 worker，表现为 GPU 占用显存但利用率为 0、所有进程 CPU 零增长。

规避手段（按效果排序）：

1. `persistent_workers=True`：worker 全程只 spawn 一次，从根上避开每 epoch 重建（推荐，尚未改入 `trainer.make_loader`）。
2. `data.num_workers=0`：完全不用子进程，最稳，但 GPU 利用率只有约 25%，单 epoch 从 14 分钟涨到 37 分钟。
3. `num_workers=2` 配合“日志静默超过 10 分钟即杀进程、从最新 last.pt 续跑”的看护脚本：实测 8 个 epoch 崩一次，自动恢复后只损失当时正在跑的批次，是最快且可接受的组合（本轮 640 即用此法）。

注意 `train --resume` 的兼容性检查逐字段比对配置，**num_workers、verify_images 等字段改动都会直接报错**，因此只能在实验开始前定好这些值；`--init-from` 新建实验则不受此限制。

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

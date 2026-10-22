# 延迟预算与解码策略

设计见 `.w1mer/detail/03_latency_plan.md`。本文只记录**实测**结果、判定门槛和提交时使用的命令。

部署约束：模型文件压缩前总量 ≤ 500 MB；NVIDIA T4 16 GB 上单图端到端 ≤ 100 ms；
提交环境只保证 PyTorch/torchvision。

## 1. 结论

- 默认 `data.decode = "full"`，与历史训练/推理**逐像素一致**；本任务不改变任何既有实验的数值。
- `"draft"` 使用 Pillow `Image.draft("RGB", (S, S))`，在 EXIF 旋转、裁剪、letterbox **之前**解码，
  保证解码后短边 ≥ S、不放大。实测 384 输入下解码 + 裁剪/letterbox 从 **454 ms 降到 30 ms**
  （p50，含裁边的 100 张 valid 随机图）。
- 端到端（V100，384 fp16，单图，20 张 valid 含 10 张最大图）：draft **p50 55.3 ms / p95 82.2 ms**；
  full **p50 870 ms / p95 1038 ms**。
- draft 的 **tensor 一致性门槛在开启黑边裁剪时边缘性 FAIL**（p95 = 0.0109 > 0.010；
  不裁剪时 0.0031 PASS）。原因是低分辨率解码使黑边 bbox 偏移 ≤0.8%，裁剪窗口错位几个像素。
  在拿到 checkpoint 并实测全 valid 集 AUROC |Δ| 之前，**不切换任何配置到 draft**
  （当前所有实验仍为 full）。
- 无 checkpoint 可用，故 AUROC 一致性与 500 MB 体积门槛本次未测，已挂到 task 09 验收。

## 2. 实测

环境：Tesla V100-SXM2-16GB，torch 2.5.1+cu121，torchvision 0.20.1，Pillow 13；
共享服务器当时 load average 12–19（其他任务同时运行），CPU 阶段数字偏保守。

### 2.1 解码耗时（384，100 张随机 valid，单图，p50/p95 ms）

`scripts/check_decode_parity.py --config configs/experiments/b0_crop384_control.json --timing`：

| 阶段 | full p50 | full p95 | draft p50 | draft p95 |
|---|---:|---:|---:|---:|
| 文件读取 | 0.63 | 9.96 | 0.63 | 9.96 |
| 解码 | 104.30 | 255.81 | 15.49 | 32.79 |
| 裁剪 + letterbox | 350.08 | 770.43 | 14.46 | 27.12 |
| **解码 + 预处理** | **454** | **1026** | **30** | **60** |

不裁剪黑边时（仅 letterbox）：full 99.70 / 253.76，draft 15.32 / 32.99；
预处理 full 38.38 / 76.55，draft 2.91 / 4.50。
开启裁剪时 full 路径要在原始 16 MP 数组上做 max-channel 扫描，因此预处理本身成为主成本。
draft 选择缩放倍数：p50 ×4，p95 ×8（100 张中全部发生缩放）。

### 2.2 端到端延迟（V100，384，fp16，batch=1，进程内，20 张 valid 含 10 张最大图）

`scripts/benchmark_latency.py --config configs/experiments/b0_crop384_control.json --images 20 --include-largest 10 --decode draft`

| 阶段 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| read | 0.66 | 31.87 | 41.21 | 43.54 |
| decode | 24.74 | 34.06 | 34.38 | 34.46 |
| preprocess | 13.59 | 26.37 | 26.39 | 26.40 |
| transfer | 0.83 | 0.92 | 0.99 | 1.01 |
| forward | 7.22 | 7.49 | 7.56 | 7.58 |
| post | 0.07 | 0.10 | 0.11 | 0.11 |
| **total** | **55.28** | **82.15** | **84.87** | **85.55** |

同配置 `--decode full`：total p50 870.13 / p95 1038.35（decode 201/250，preprocess 659/778）。
冷启动第一张图（CUDA context + cuDNN autotune + 首次前向）：**draft 2.8 s**。
若评判把第一张图计入 100 ms，必须先做 warmup 或预创建 CUDA context。

### 2.3 解码一致性（384，100 张随机 valid）

| 指标 | 不裁剪 | 裁剪（部署配置） | 门槛 |
|---|---:|---:|---:|
| tensor 每图平均 \|Δ\| p50 | 0.00073 | 0.00364 | — |
| tensor 每图平均 \|Δ\| p95 | **0.00312** | **0.01092** | ≤ 0.010 |
| tensor 每图平均 \|Δ\| max | 0.0165 | 0.0165 | — |
| 元素级 \|Δ\|（各图 p99 的最大值） | 0.057 | 0.227 | — |
| 元素级 \|Δ\| max | 0.573 | 2.000 | — |
| bbox 归一化最大偏移 | — | 0.82% | ≤ 1% |
| AUROC full / draft | 待 checkpoint | 待 checkpoint | \|Δ\| ≤ 0.001 |

裁剪配置下 100 张中有 6 张每图平均 |Δ| > 0.010，全部来自 ×4–×8 缩放的大图。
**结论：draft 的解码质量本身可以接受（不裁剪路径稳健 PASS），但与黑边裁剪组合时
处于门槛边缘；是否可部署必须等 checkpoint 的全 valid AUROC 判定。**

## 3. T4 外推与验收

服务器只有 V100。`benchmark_latency.py` 对非 T4 的 CUDA 设备输出 ×2.0 的 forward 估计，
CPU 阶段按原值计入（解码/预处理是 CPU 绑定的，不能乘 GPU 系数）：

- 投影 total p50 **62.6 ms**，p95 **89.7 ms**（forward p50 7.2 → 14.4 ms）。
- 若评判机 CPU 比服务器慢 50%，p95 约 **+20 ms ≈ 108 ms —— 会超限**。
  因此 T4 上的真实测量不可省略，且需记录评判机 CPU 情况。
- **这不是 T4 测量结果。** 最终验收必须在 T4 上原样运行同一脚本：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/benchmark_latency.py \
  --checkpoint outputs/runs/<run>/best.pt \
  --data-root /path/to/glaucoma_first --split valid \
  --images 200 --include-largest 10 --decode draft --output /tmp/latency_t4.json
```

判定：`TOTAL p95 <= 100 ms` 且 checkpoint 文件 ≤ 500 MB。只有 `best.pt` 计入提交体积，
`last.pt`（含优化器状态，约 3×）绝不提交。

## 4. 提交时使用的命令

CPU 侧（无需 GPU）：

```bash
PYTHONPATH=src python scripts/check_decode_parity.py \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --split valid --samples 200 --image-size 384 --timing

PYTHONPATH=src python scripts/benchmark_latency.py \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --split valid --images 3 --cpu-only --no-enforce
```

有 checkpoint 后的一致性判定（全 valid 集，约 2–4 min GPU）：

```bash
PYTHONPATH=src python scripts/check_decode_parity.py \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --split valid --samples 200 --checkpoint outputs/runs/<run>/best.pt \
  --decode-compare draft --device cuda --output /tmp/parity.json
```

## 5. 训练/推理一致性

- `decode` 是 `DataConfig` 字段，写入 `config.json`、`metadata.json`、缓存 manifest 和
  `best.pt`。`evaluate` / `predict` 只读 checkpoint 内的值，没有独立默认值。
- 缓存 manifest v3 记录 decode；v1/v2 视为 `full`，仍可加载。`draft` 缓存与 `full`
  配置互相拒绝（必须从原图重建，不能由旧缓存转换）。
- 当前所有实验配置均未切换到 `draft`。切换是 task 09 全量重训时的一行配置改动，
  且必须先通过第 2.3 节的判定。

## 6. 集成/TTA 的延迟门槛

单次 384 fp16 forward 的实测值见 2.2。按 100 ms 总预算，第二份模型或 TTA 视图会使
`forward` 翻倍；是否可行必须用 `benchmark_latency.py` 实测**该具体组合**，不能只看
500 MB 体积限制。

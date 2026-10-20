# itgpu 训练提速

2026-10-17 实测：运行配置已经是 V100、384×384、batch=64、梯度累积=2、workers=8、FP16。B0/B1 分别使用独立 GPU；原始数据位于 NFS。数据加载子进程持续占满 CPU，GPU 利用率在 0% 和 99% 之间波动。

## 已验证的改进

将 RGB 解码、EXIF 方向处理和确定性 Letterbox 的输出保存为本地磁盘上的 uint8 `.npy`，训练时仍逐轮执行原有随机增强。原始图片、划分、分辨率、有效 batch、优化器和学习率计划不变。

| 1024 张训练图，单张空闲 V100，相同配置 | 耗时 | 吞吐 |
|---|---:|---:|
| 原始读取及预处理 | 40.723 秒 | 25.15 张/秒 |
| 本地无损缓存 | 12.634 秒 | 81.05 张/秒 |

该短跑为 3.22 倍吞吐；包含 DataLoader 启动，不能直接当作完整 epoch 的加速比。两次短跑损失相同，32 张真实图的随机增强张量逐元素完全一致。合成测试还覆盖缓存与原流程一致性、源文件变化拒绝，以及使用缓存恢复后的权重与完整训练完全一致。

## 使用

先选择空间足够的本地磁盘。train/valid 共 57,245 张图，384 缓存约占 24 GiB。不要把缓存放到 NFS，也不要占用内存盘 `/tmp` 或 `/dev/shm`。

```bash
python -m glaucoma cache-images \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --output /var/tmp/自己的独立缓存目录 --workers 8

CUDA_VISIBLE_DEVICES=空闲卡号 python -m glaucoma train \
  --config configs/baseline_b0.json --device cuda:0 \
  --resume 原运行目录/last.pt --image-cache /var/tmp/自己的独立缓存目录
```

缓存仅在全量构建成功后生成 `manifest.json`。训练入口检查图片尺寸、CSV 哈希和源文件大小/修改时间；任何不一致都会报错。启用图片审计时检查缓存中的解码结果，加载损坏项仍报错，不跳过样本。缓存参数属于运行时 I/O 设置，支持原配置的断点恢复。

复测吞吐需选择空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=空闲卡号 OMP_NUM_THREADS=2 python scripts/benchmark_training_io.py \
  --config configs/baseline_b0.json --cache /var/tmp/自己的独立缓存目录 \
  --samples 1024 --output outputs/新的测速文件.json
```

`scripts/switch_cached_training.py` 是 Linux 迁移工具：等待缓存完成并核验测速收益，在旧任务下一份完整 checkpoint 保存后暂停旧进程，用空闲卡恢复；新任务完成 100 个 batch 后才释放旧进程。启动失败时恢复旧进程。它检查进程所有者、启动时间、工作目录和配置，并保留原输出和权重。迁移状态和新训练日志应一起保存，不能仅凭启动成功认定迁移成功。

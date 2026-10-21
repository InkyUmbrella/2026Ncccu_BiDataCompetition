# 年龄残差微调

模型 `convnext_tiny_age_residual` 保留 B0 图像编码器和分类头，在原始 logit 上增加年龄修正：

```text
logit = image_logit + age_residual(age_z, is_missing, is_invalid)
```

年龄沿用训练集统计进行标准化；缺失值和超出 `(0, 100]` 的年龄使用训练集年龄中位数填充，并提供独立标记。年龄网络为 `3 → 16 → 1`，最后一层权重和偏置初始化为零。从纯图像 B0 加载后，在 eval 模式下初始预测与原模型一致。

## 模型接口

通过模型注册表以 `model.name = "convnext_tiny_age_residual"` 构建，构造参数为 `hidden_dim` 和 `dropout`。统一实现 `forward(image, age) -> logits`：图像为 `[N, 3, H, W]`，年龄特征为 `[N, 3]`，输出为 `[N]` 原始 logit，由现有评估和推理流程计算概率。模型保留 B0 的 `encoder.*`、`head.*` 参数名，新增 `age_residual.*`，通过 `set_image_frozen(bool)` 控制图像分支是否参与训练。训练配置中的 `freeze_image_epochs` 和 `image_lr_scale` 分别控制冻结时长和图像学习率比例。

## 默认实验

配置：`configs/experiments/b0_crop384_age_residual.json`。使用 crop384 的最佳 `best.pt` 初始化，共训练 20 轮，batch 64、梯度累积 2，年龄分支学习率 `1e-4`，图像分支学习率为其 0.1 倍。前 3 轮冻结图像编码器和分类头，同时关闭图像分支的 Dropout 和 stochastic depth；从第 4 轮起联合微调。

以下命令在项目根目录运行，`SOURCE_BEST_PT` 应替换为已训练好的纯图像 crop384 模型的 `best.pt` 路径：

```bash
PYTHONPATH=src python -m glaucoma train \
  --config configs/experiments/b0_crop384_age_residual.json \
  --init-from SOURCE_BEST_PT \
  --data-root dataset/glaucoma_first
```

使用已有的 crop384 缓存时追加 `--image-cache CACHE_CROP384`。缓存必须与配置的分辨率、裁边阈值和边距一致；程序会拒绝不匹配的缓存。没有缓存时直接从原图裁边、缩放并执行随机增强。

需要新建缓存时：

```bash
PYTHONPATH=src python -m glaucoma cache-images \
  --config configs/experiments/b0_crop384_age_residual.json \
  --data-root dataset/glaucoma_first \
  --output outputs/cache_crop384 --workers 8
```

`--init-from best.pt` 新建实验并重置优化器、学习率计划、训练历史和随机状态，保留来源 checkpoint 的年龄统计。支持同结构模型初始化，以及 B0 转入年龄残差；旧 B1 的拼接融合权重不能直接转入。

`--resume last.pt` 恢复原实验的优化器、随机状态、历史和冻结阶段，配置只允许改变数据根目录与输出根目录。两种入口互斥。冻结图像分支的实验必须提供其中一种入口，避免冻结随机图像权重。

## 评估

现有 `evaluate` 和 `predict` 自动从 checkpoint 恢复模型、年龄统计和裁边参数，命令用法不变。`history.json` 增加 `image_frozen` 标记，`metadata.json` 记录初始化权重路径。

年龄是否有效需通过验证集比较确认：以 crop384 为基线，优先比较 AUROC，再看 AUPRC 和各年龄组表现。还应增加纯图像模型的匹配微调对照，保证图像分支的更新轮数、学习率和冻结日程相同，以区分年龄信息与额外训练的收益。本次实现不代表已完成效果验证。

## 本地验证

```powershell
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '2'
.venv/Scripts/python.exe -B -c "import torch, unittest; torch.set_num_threads(2); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not result.wasSuccessful())"
```

测试包含真实 ConvNeXt 的 B0 初始预测一致性、冻结期间图像权重不变且年龄权重更新、差异学习率、跨冻结阶段断点恢复、裁边缓存一致性及推理输出。

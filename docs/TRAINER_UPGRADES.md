# 训练增强：EMA、随机深度、Mixup/CutMix、标签平滑

四项增强全部**默认关闭**。不填写新配置项时，训练数学与随机数消耗和旧版本一致，旧配置与旧断点仍可直接加载/恢复。启用任一项后，配置会写入 checkpoint，恢复训练时会校验配置一致（仅 `data.root` 与 `output_root` 可改）。

## 配置项

```json
{
  "model": {
    "name": "convnext_tiny",
    "params": {"use_age": false, "hidden_dim": 128, "dropout": 0.2, "stochastic_depth_prob": 0.1}
  },
  "train": {
    "ema_decay": 0.9998,
    "ema_warmup_steps": 2000,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "label_smoothing": 0.05
  }
}
```

| 配置键 | 默认 | 语义 |
|---|---|---|
| `train.ema_decay` | `null` | `(0,1)` 内的浮点数；`null` 表示关闭。所有浮点参数与 buffer 维护 fp32 影子副本，每次真实优化器 step 后更新。 |
| `train.ema_warmup_steps` | `0` | `>0` 时衰减系数为 `min(decay, (1+step)/(10+step))`；`0` 表示固定 `ema_decay`。 |
| `model.params.stochastic_depth_prob` | `0.0` | `[0,1]`，直接传给 `torchvision.models.convnext_tiny` 的随机深度概率。B0/B1 都支持。 |
| `train.mixup_alpha` | `null` | `>0` 的 Beta 参数；`null` 关闭 Mixup。 |
| `train.cutmix_alpha` | `null` | `>0` 的 Beta 参数；`null` 关闭 CutMix。两者同时启用时每 batch 以 50/50 概率二选一。 |
| `train.label_smoothing` | `0.0` | `[0,0.5]`；目标变为 `y*(1-s) + 0.5*s`。 |

## EMA

- 影子副本从模型**初始化之后**的权重复制，因此不改变初始化、优化器状态或随机数。
- AMP 因梯度溢出跳过 optimizer step 时，EMA 同步跳过（与 `global_step` 的判定一致）。
- 启用后，每个 epoch 的验证、最佳模型选择以及 `best.pt`（推理文件）都使用 EMA 权重；`last.pt` 仍保存当前训练权重，同时额外保存 EMA 影子副本与更新计数，`--resume` 时精确恢复。`metrics.json` 中的最终指标同样基于 EMA 权重。
- `--init-from` 不使用来源 checkpoint 的 EMA，EMA 从新实验当前权重开始。

## 随机深度

`stochastic_depth_prob` 属于 `model.params`，会随 checkpoint 保存，推理时自动构造相同结构。`0.0` 与不写该键完全等价（同样的初始化权重与 state_dict）。训练时逐 block 随机丢弃残差路径，评估时恒等。

## Mixup / CutMix

- 每个 batch 在样本维随机置换后混合：Mixup 使用 `lam ~ Beta(alpha, alpha)` 混合图像与目标；CutMix 按 `lam` 生成矩形区域并用另一张图替换，目标按实际替换面积 `1 - (w*h)/(W*H)` 混合。
- 目标为软标签（浮点），BCE 损失直接接受。**`pos_weight` 语义**：损失为 `-(w * y * log σ(x) + (1-y) * log(1-σ(x)))`，软标签 `y∈[0,1]` 使正类权重按 `y` 比例作用于正类项，而不是按样本数计数；因此 Mixup 下 `pos_weight` 的有效强度会随软标签缩小，调参时需注意。
- 随机数全部来自当前进程的全局 RNG（`seed_everything` 设定、`last.pt` 恢复）。关闭时**不消耗任何随机数**；开启后同一 seed、同一 epoch 边界恢复可复现。
- 标签平滑先作用于原始硬标签，再做 Mixup/CutMix，因此混合两侧的目标都已平滑。

## 兼容性

- 新增字段在 `TrainConfig` 中带默认值，旧 JSON 配置与旧 checkpoint 的 `config` 缺少这些键时自动补默认值，`to_dict()/from_dict()` 往返稳定。
- 旧断点（记录中无 `ema` 字段）只能以 `ema_decay: null` 恢复；用 EMA 配置恢复旧断点会明确报错。
- `stochastic_depth_prob` 不新增参数，旧 ConvNeXt checkpoint 仍可 `strict=True` 加载。

## 测试

`tests/test_upgrades.py` 覆盖：EMA 更新与 warmup 公式、AMP 跳过、断点恢复逐位一致；Mixup/CutMix 软标签与确定性、`pos_weight` 软标签语义；标签平滑目标与损失；随机深度接线及 eval 恒等；默认配置下与旧训练循环逐位一致且不多消耗随机数。

GPU 冒烟（合成数据、ConvNeXt-Tiny、64px、FP16、EMA + Mixup/CutMix + 标签平滑 + 随机深度，2 个 epoch）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/smoke_upgrades.py
```

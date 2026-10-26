# 2026-10-23：以 AUROC 为目标的第二轮微调

四项正式实验和补取的历史材料统一见 [训练结果汇总](TRAINING_RESULTS.md)；含完整日志、实际源码、初始化/断点哈希、验证预测和无法补回的逐轮checkpoint缺口。

两项已正常完成（10/15轮），旧任务已停止跟踪。验证结果：refine 第1轮 raw，AUROC 0.9873283254、AP 0.8322199715、PR梯形面积 0.8319153177；restart 第6轮 raw，AUROC 0.9831924958、AP 0.8028904289、PR梯形面积 0.8024768469。refine 相对B0的 AUROC 差值为 +0.0008230944，1000次配对分层 bootstrap 95% 区间 [-0.000831978, 0.002186613]，仍跨0。旧B0和各实验权重均保留。

后续队友训练使用 [4090交接文档](TRAINING_4090.md) 中的通用启动器。下文保留原服务器实验设计；`scripts/pretrained_improve.py` 是绑定原实验上下文的历史监督入口，不作为新机器的训练入口。

第一轮B0的最佳验证AUROC为0.9865052309753692（第5轮）；第30轮训练损失降至0.006327，验证AUROC却为0.981719，提示后期过拟合。年龄残差训练20轮未提高AUROC，因此本轮集中调整纯图像微调，保留原B0为独立基准。

| 实验 | 初始化 | 轮数 | 分类头最高学习率 | 骨干最高学习率 | EMA |
|---|---|---:|---:|---:|---:|
| b0_crop384_refine_ema | 固定的B0第5轮best | 10 | 1e-5 | 1e-5 | 0.999 |
| b0_crop384_imagenet_backbone_ema | 官方ImageNet-1K V1 | 15 | 1e-4 | 2e-5 | 0.999 |

其余保持384裁黑边/等比缩放补边、ImageNet归一化、seed42、AdamW、pos_weight=5、weight_decay=0.05、1轮warmup、余弦衰减、batch32×累积4。使用物理GPU4、5，启动前同时核查进程与显存，GPU0–3始终留给队友。独立服务器目录为 `/home/lichunyu/competition-pretrained-improve`，不改动前两次实验目录。

`backbone_lr_scale` 只额外缩放 `encoder.*` 的学习率，旧 `image_lr_scale` 对年龄模型的语义不变。两者默认1，因此旧配置/完整断点继续兼容。EMA默认关闭，仅在真实成功的优化器更新后推进；AMP跳过更新时不推进。每轮同一训练轨迹同时评估raw与EMA，保存各自指标，按原有AUROC/PR梯形面积选模。最佳推理权重仍是普通单模型；不增加推理次数。完整断点新增EMA权重及累计更新数，恢复时必须一并恢复。

这两组是有限的配方尝试，不是分离每个超参数因果贡献的完整析因实验。完整官方valid用于选模和报告，标签、划分及指标定义不变；不使用test。相对旧B0的改进需由真实验证确认，不能把工程测试或微小单次差值当作确定收益。最终额外报告同一验证样本上的配对分层bootstrap AUROC差值区间（1000次、固定随机种子），该区间不消除多次选模带来的偏差。

复用已完成的57,245张无损缓存审计时，入口逐项核对原成功审计、缓存manifest和官方两个CSV的SHA256及51396/5849计数，并保存audit_reuse来源；正常ImageCache仍核查源文件签名、缓存数组形状与dtype，不跳过图片。

先通过回归测试，再执行真实GPU入口。每次启动使用新输出目录，记录PID/start_ticks、CUDA映射、配置/源码哈希、初始化SHA256、日志及退出码。父流程完成后读取best/last、检查有限性/EMA更新数/完整恢复状态，核验5849张预测名称顺序和191阳性，重算AUROC/AP/PR梯形面积。不覆盖原B0，任何新配方变差时仍保留原最佳模型。

参考：[ConvNeXt官方训练与微调配方](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md)、[官方分层学习率实现](https://github.com/facebookresearch/ConvNeXt/blob/main/optim_factory.py)、[PyTorch关于EMA的配方说明](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)。这些资料支持尝试方向，不能保证在本青光眼验证集上改善AUROC。

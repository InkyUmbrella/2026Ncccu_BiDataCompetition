# shimmer 训练交接

最新0.988527结果的分析、下一轮优先级和运行命令见 [第二轮优化](round2/README.md)。

源代码基于服务器 `heyuhang/w1mer-opt` 的 `21e9e09`，保留原来的科研代码、配置和历史。预训练接入参考 `InkyUmbrella/2026Ncccu_BiDataCompetition` 的 `codex/pretrained-auroc-improve`（`1afe954`），并保留原分支的 stochastic depth、draft 解码、采样、Mixup、标签平滑和 EMA 实现。用户已确认赛事允许外部预训练权重。

## 已有成绩与证据

服务器实际记录：512 seed43 的最佳 AUROC **0.9769838934**、AP **0.6460334970**，metrics.best_epoch=37（从0计数，即第38轮），完成60轮。截图第35轮的0.9766是训练中间成绩。512 seed44 最佳 **0.9623342938**，seed42 **0.9682796356**，不能假定双模型融合一定更好。seed43 来自恢复运行；前序和恢复运行的 config/metadata/history 已一并保存。

`results.json` 索引正式完成实验；`evidence/` 保留服务器原始 JSON，`SHA256.json` 可校验。未上传原图、病人逐样本资料、模型权重。原路径、环境及数据CSV哈希用于复现。这里保留的是实验记录，未重新运行全验证集，因此不将指标检查当作独立复评。

预训练源分支的 docs/TRAINING_RESULTS.md 报告：384 ImageNet B0=0.986505231，低学习率 refine=0.987328325。它是目前已找到的比较目标，不代表审计了所有分支或已复现该成绩。

本地 RTX 4090 D 复跑 512 预训练配方（`shimmer_512_pretrained_reference`，seed42）：第 23 轮达最佳，全验证集重评为 AUROC **0.9885271098**、AP **0.8275141457**、AUPRC 梯形面积 **0.8272453111**，Sensitivity@Sp95 = 0.9476。实际完成 **28/30 轮**——第 23 轮后连续 5 轮未超越，按最佳轮次提前停止，未跑满 30 轮，因此该数字与服务器 60 轮结果不完全可比。训练期间因外部进程终止自动续跑 3 次，每次均自 `last.pt` 恢复（history 连续、无重置）；最终 best.pt 及其评估来自原始配置运行，未改动任何超参数。该结果高于预训练源分支报告的 0.9873，但仅单 seed、单机复现，不作为稳定结论。证据：`evidence/outputs/runs/local_20261027T123315Z/`。

## 实现与实验选择

- 固定 torchvision ConvNeXt-Tiny `IMAGENET1K_V1`，ImageNet 分类器丢弃，比赛二分类头重新初始化。保留完整视野的裁黑边/letterbox，不使用 ImageNet 中心裁剪。
- ImageNet mean/std 写入配置和checkpoint；缓存仍是 RGB uint8，归一化在增强后执行，训练、评估、预测一致。
- 只有从头启动预训练微调时加载官方权重。resume、init-from、init-encoder、predict/evaluate 不重新下载。旧配置默认对称归一化和随机初始化，旧checkpoint保持兼容。
- `backbone_lr_scale` 只缩放encoder；旧 `image_lr_scale` 语义保留。新512融合配方默认骨干LR为分类头的0.2倍；该参数是待验证尝试，源分支低骨干LR实验反而低于其B0，不能认为它必然提高分数。
- `evaluate_raw_with_ema=true` 每轮比较raw与EMA，以AUROC/PR梯形面积选best；输出记录权重来源。best.pt仍是一个模型，不增加推理次数。旧配置默认关闭，继续只评估EMA。
- 保留512 seed43的采样/Mixup/标签平滑，同时提供 `pretrained_reference` 配方：自然采样、pos_weight=5、不加Mixup/平滑/EMA，以防旧随机训练的强正则限制预训练收益。
- 16×8有效batch=128适合先在16GB卡试运行，和原32×4有不同的Mixup批内组合，不能声称逐位复现。384对照保持32×4。

## 新服务器运行

使用已有可用的PyTorch/torchvision环境，项目根目录执行。数据目录须包含train.csv、valid.csv、test.csv及图片子目录；只用比赛train训练，valid选模。

```bash
export PYTHONPATH=src
# 第一优先：与上游预训练成功配方对应的512版本
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train \
  --config configs/experiments/shimmer_512_pretrained_reference.json \
  --data-root /实际路径/glaucoma_first

# 第二优先：融合原512 seed43配方，评估raw和EMA
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train \
  --config configs/experiments/shimmer_512_imagenet_seed43.json \
  --data-root /实际路径/glaucoma_first
```

首次检查可追加 `--stop-after-epochs 1`，之后用相同配置 `--resume /该次运行/last.pt`，保持原30轮调度。新实验输出目录自动唯一生成。两条训练顺序运行；不要在同一GPU同时启动。显存不足则同时减batch_size、增accumulation_steps保持128，但会改变Mixup批内分布。seed44配置作为后续稳定性对照，不必先全部训练。

若需要定位分辨率收益，运行 `shimmer_384_pretrained_reference.json`。它使用源分支384/full解码的主要配方，但硬件、环境差异可能影响结果。

若512融合配方已有好的best，可低学习率微调（配置模型和归一化必须相同）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train \
  --config configs/experiments/shimmer_512_refine_seed43.json \
  --data-root /实际路径/glaucoma_first --init-from /融合实验/best.pt
```

该refine配置用于 `shimmer_512_imagenet_seed43`；不要将旧对称归一化0.97698模型直接传入，因为预处理不兼容。也不要把ImageNet骨干覆盖到该旧模型里。

可选无损缓存（约43 GiB以上，另外预留源图、checkpoint和临时空间）：

```bash
python -m glaucoma cache-images \
  --config configs/experiments/shimmer_512_imagenet_seed43.json \
  --data-root /实际路径/glaucoma_first --output /本地SSD/cache512 --workers 4
```

训练追加 `--image-cache /本地SSD/cache512`。384/full与512/draft缓存不能混用。默认不建缓存即可训练。

生成提交：

```bash
CUDA_VISIBLE_DEVICES=0 python -m glaucoma predict \
  --checkpoint /最佳运行/best.pt --data-root /实际路径/glaucoma_first \
  --output outputs/submission.csv
```

输出UTF-8，列为Image_name,Probability,Pred_Label，阈值0.5。AUROC由概率排序决定，调整阈值不会提升AUROC。正式test应有5855行；提交文件已存在时入口拒绝覆盖。模型大小及T4单张端到端100ms限制仍需在目标硬件实测；V100测试不代替T4验收。保留单模型部署优先，不预设两模型/TTA满足延迟。

## 验证

合成测试不下载外部权重、不使用比赛图像、不启动GPU训练：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=src python -c \
 'import torch,unittest;torch.set_num_threads(2);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"));raise SystemExit(not r.wasSuccessful())'
```

具体执行结果见 validation.txt。

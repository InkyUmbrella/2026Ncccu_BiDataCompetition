# 2026-10-26 服务器交接验证

本次只在服务器 `glaucoma` 环境执行测试。本机仅编辑、打包和操作 Git。未在服务器启动新的正式数据长训练，未生成或提交比赛预测。

独立测试目录（旧实验目录保持原状）：

```text
/home/lichunyu/competition-training-handoff/validation_20261026T143528Z
```

部署前对 56 个源码/配置/文档文件逐文件核对 SHA256，原始清单为该目录的 `deployment_manifest.json`。最终交接文档在测试后补齐；运行代码与测试代码保持已测试内容。Windows 工作区与 Git blob 的文本行尾归一化差异按 LF 比对。

## 环境

| 项目 | 实际版本 |
|---|---|
| Python | 3.11.17 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| CUDA runtime / cuDNN | 12.1 / 90100 |
| 测试硬件 | Tesla V100-SXM2-16GB，物理 GPU5 |

GPU 测试启动前确认 GPU5 无计算进程、显存占用为0；使用 `CUDA_VISIBLE_DEVICES=5`，没有使用 GPU0–3 或其他用户的 GPU6/7。4090 硬件和 Windows 训练入口尚未实测。

## 结果与证据

下表所有路径相对于上述测试目录。

| 检查 | 结果 | 日志/报告 |
|---|---|---|
| CPU 全回归 | 35项，34通过、1项 opt-in CUDA 跳过，48.223秒，退出0 | `evidence/cpu.log`、`cpu.exit` |
| B0/B1 实际 ConvNeXt FP16 | 两种变体均完成真实优化器更新，384/batch16，退出0 | `evidence/gpu_fp16.log`、`gpu_fp16.exit` |
| 启动器→部分训练→完整恢复→核验 | 合成256 train/8 valid，384、batch32×累积4、EMA0.999、官方预训练，1轮+恢复1轮，退出均0 | `evidence/gpu_partial.log`、`gpu_resume.log`、`gpu_report.json` |
| 包构建 | pip wheel 成功，退出0，不改服务器全局环境 | `evidence/package-build.log`、`package-build.exit` |

CPU 测试覆盖逐轮不可覆盖的文件留存、初始状态/源码归档、篡改检测、失败启动日志/退出码；EMA/优化器/RNG 恢复与不中断运行的精确比较，以及原有缓存、预训练迁移和推理回归。

GPU 完整入口的 `global_step` 从2恢复到4，最终 `EMA n_averaged=4`。父/子运行的逐轮记录分别为 `[1]` 和 `[2]`；各自16个归档文件均重算 SHA256 通过。核对了8张验证预测的数量与名称顺序、有限权重、完整恢复字段、父断点初始化哈希及两个启动日志的哈希。合成数据的指标只验证计算流程，不代表正式验证 AUROC。

对应运行目录：

```text
evidence/gpu_runs/synthetic_audit_gpu/seed42_20261026T143746Z_fe2910a1
evidence/gpu_runs/synthetic_audit_gpu/seed42_20261026T143810Z_0ac872db
```

完整控制台日志及退出码另存于：

```text
evidence/gpu_runs/launches/20261026T143741Z_1d5a17fe/
evidence/gpu_runs/launches/20261026T143804Z_9f276eb4/
```

`evidence/validation_summary.json` 汇总测试范围、路径及证据文件 SHA256。主要日志哈希：

```text
cpu.log         f86dea587d3d32029072538be4bfbd0683d282b84280eb14949510e9dcb55674
gpu_fp16.log    48344505df9c41a3c5eee1b1bdfbb3c8ec3aeceb10e399ac493f7ec408fd1868
gpu_partial.log eced657091e1531f075c21312ada8345b67e2687dff2186ec0551b9cc40f108b
gpu_resume.log  9cdb5a3aa218fff2338d833e1370654a14ca00fca2cdcae11ad563b512d043f9
```

正式训练启动和恢复命令见 [4090交接](TRAINING_4090.md)。旧实验没有逐轮留存的 checkpoint 不在本次验证中补造。

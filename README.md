# 青光眼眼底图像分类

基线包括 B0（ConvNeXt-Tiny 整图分类）和 B1（相同骨干融合年龄），均为随机初始化。
实验规格见 [基线设计](docs/BASELINE_DESIGN.md)，通信环境见 [六卡测试报告](docs/SIX_GPU_FEASIBILITY.md)。

队友在 RTX 4090 上继续训练，请使用 [4090 训练交接](docs/TRAINING_4090.md)：包含固定依赖、B0→refine 启动、完整断点恢复和赛事核验材料保存方法。

已完成的B0、年龄残差、refine和restart结果及历史材料，见 [训练结果汇总](docs/TRAINING_RESULTS.md) 和 [归档材料目录](training_records/20261022-23/README.md)。当前最高完整valid AUROC为refine的0.987328325；历史逐轮checkpoint缺口在报告中明确列出。

后续开发以 [参赛规则合规核查](docs/COMPLIANCE_REVIEW.md) 为边界，按 [T4交付与AUROC开发计划](docs/DEVELOPMENT_PLAN.md) 推进。[官网](https://www.ncccu.org.cn/index/Paper/case2.html?way=1)确认区域赛/省赛截止2026-11-06 23:59（北京时间），最终评测为单张T4 16GB、Python 3.12/PyTorch 2.10.0/CUDA 12.8。**当前工作区已写入DINOv2、512配置及统一预测/计时入口，尚未运行验证；用户已授权将本批源码推送GitHub，服务器或4090运行位置仍等待用户指定。** 状态和待执行命令见 [本批实现说明](docs/DINOV2_IMPLEMENTATION.md)。公开数据记录的访问范围由用户稍后自行改私有，完成状态尚未核实。

## 目录

```text
src/glaucoma/
  cli.py                 # check-data / train / evaluate / predict / models
  config.py              # 严格配置与参数校验
  data/                  # CSV、图片、年龄预处理、增强、数据审计
  models/                # 模型注册与各模型实现
  engine/                # 训练、评估、优化器、断点保存与恢复
  metrics.py             # AUROC、PR 指标、年龄分层
  utils.py               # 随机种子、运行记录、文件保存
configs/                 # B0、B1 独立 JSON 配置
tests/                   # 合成数据测试
scripts/                 # 已有服务器诊断脚本
outputs/                 # 每次运行的独立目录，不提交 Git
```

## 环境与数据

服务器项目路径为 `/home/lichunyu/competition`，数据路径为 `dataset/glaucoma_first/`。
以下命令在服务器 Bash 和项目根目录中运行。已有 `glaucoma` 环境包含 PyTorch 2.5.1、torchvision 0.20.1，基线无需 timm。

```bash
conda activate glaucoma
python -m pip install -e . --no-deps --no-build-isolation
python -m glaucoma --help
```

也可以不安装项目，使用 `PYTHONPATH=src python -m glaucoma ...`。依赖声明在 `pyproject.toml`；不要为了运行本项目重装现有 CUDA/PyTorch 环境。

数据应包含 `train.csv`、`valid.csv`、`test.csv` 及各自图片目录。CSV 要求 `Image_name,Age`，训练和验证另有 `True_Label`（0/1）。读取器自动匹配 `.jpg/.JPG`，保存预测时保留 CSV 中的原始名称与顺序。

## 检查数据

```bash
python -m glaucoma check-data --data-root dataset/glaucoma_first \
  --hashes --output outputs/data_audit.json
```

该命令逐张解码；`--hashes` 额外检测文件字节完全相同的重复图。它不能识别重压缩图片、同一患者双眼或所有近似重复。发现问题会输出证据，不自动改变划分。报告路径需尚不存在。训练入口默认再次检查 train/valid 解码。

## 训练

下面是队友自有4090机器的历史基线入口示例，按实际设备编号替换；不代表需要重新训练旧任务。当前共享服务器只做工程验证，仅允许检查空闲后的物理GPU4/5，GPU0–3、6/7不可使用。后续正式训练优先使用上方交接文档中的审计启动器。

```bash
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train --config configs/baseline_b0.json
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train --config configs/baseline_b1.json
```

这是单卡入口；自有单卡机器上两条命令顺序运行，不要用 torchrun 启动它。路径默认相对当前工作目录，支持 `--data-root`、`--output-root` 和 `--device` 覆盖。

首次可使用 `--stop-after-epochs 1` 检查一轮流程，它保留原定 60 轮学习率计划。新一次正式训练应重新初始化；需要继续该实验时使用断点恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train --config configs/baseline_b0.json \
  --resume outputs/baseline/b0_image/seed42_RUN/last.pt
```

恢复从完整 epoch 的边界开始，创建新的运行目录，并恢复优化器、GradScaler、随机状态、历史记录和最佳权重。只允许修改数据路径和输出路径，CSV 内容必须一致；图片内容仍需由数据管理保证不变。相同环境与配置下可复现实验，不承诺跨硬件逐位一致。

每次运行保存配置、环境、源码快照、实际初始化及哈希、每轮完整 `checkpoints/epoch_NNNN.pt` 和时间/哈希清单，以及 `history.json`、`metrics.json`、`valid_predictions.csv`、`best.pt`、`last.pt`。`best.pt` 是推理文件；`last.pt` 包含恢复训练所需状态。使用 `python scripts/train_audited.py --config CONFIG --data-root DATA_ROOT` 自动留存完整终端日志、命令和退出码；结束后用 `python -m glaucoma verify-run --run-dir RUN_DIRECTORY` 重算产物哈希。详细目录和恢复证据链见上述交接文档。

## 评估与生成提交

```bash
python -m glaucoma evaluate --checkpoint outputs/baseline/b0_image/seed42_RUN/best.pt \
  --data-root dataset/glaucoma_first --output-dir outputs/evaluation_RUN

python -m glaucoma predict --checkpoint outputs/baseline/b0_image/seed42_RUN/best.pt \
  --data-root dataset/glaucoma_first --output outputs/submission_RUN.csv
```

模型结构、图像尺寸及年龄统计量从 checkpoint 恢复。评估目录和提交文件不能已存在。提交列为 `Image_name,Probability,Pred_Label`，阈值为 0.5；提交前按赛事要求命名为 `submission.csv`。结果行数由 test.csv 决定，正式数据应为 5,855 行。

AUROC 是选模主指标；`auprc_trapz` 与 `ap` 分开报告，避免将不同 PR 积分方式混用。模型的 T4 端到端延迟和实际比赛得分需要后续实测。

## 添加模型

1. 在 `src/glaucoma/models/` 新建模块，用 `@register_model("模型名")` 注册工厂或类。
2. 工厂接收配置 `model.params` 中的关键字参数，并返回 `nn.Module`。
3. 模型实现 `forward(image, age)`：输入分别是 `[N,3,H,W]` 和 `[N,3]`，返回 `[N]` 原始 logits。纯图像模型可忽略 age。
4. 在 `models/__init__.py` 导入模块，在新实验配置里设置 `model.name` 和 `model.params`。

训练器、评估器和预测入口无需修改。若新模型需要不同的输入模态或归一化方式，应扩展数据配置与变换，并确保相关参数随 checkpoint 保存；不要在推理代码中另写一套预处理。

## 测试

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=src python -m unittest discover -s tests -v
```

测试只使用临时合成图片，覆盖大小写匹配、年龄处理、损坏文件、真实 ConvNeXt 两个变体、梯度累积、断点恢复及评估/提交流程。测试中的小模型仅在测试进程注册，不属于正式模型列表。

默认跳过 GPU 测试。当前共享服务器确认物理GPU4的计算进程及显存均空闲后，可验证两个模型在384分辨率、batch=16下的FP16更新（也可选择确认空闲的物理GPU5）：

```bash
CUDA_VISIBLE_DEVICES=4 GLAUCOMA_CUDA_SMOKE=1 OMP_NUM_THREADS=2 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_cuda_smoke.py -v
```

2026-10-16 已在 itgpu 的 glaucoma 环境通过 9 项 CPU 合成测试，以及 B0/B1 的 V100 FP16 测试。
384×384、batch=16 时，两个变体的峰值已分配张量显存约 3,029 MiB（不含 CUDA 驱动及缓存保留开销）。
这些检查验证工程流程，不代表正式数据上的模型效果或 T4 推理延迟。后续正式实验结果及本次交接验证范围见 [改进实验](docs/PRETRAINED_IMPROVEMENT.md) 和 [服务器验证记录](docs/SERVER_HANDOFF_VALIDATION.md)。

Training I/O speedup: see [docs/TRAINING_SPEED.md](docs/TRAINING_SPEED.md).

年龄残差方案与微调命令见 [docs/AGE_RESIDUAL.md](docs/AGE_RESIDUAL.md)。

ImageNet-1K 预训练微调、同配方随机初始化对照及服务器运行方法见 [docs/PRETRAINED.md](docs/PRETRAINED.md)。原有基线配置保持随机初始化。

fwq 模型集成、B0 快照验证与并行年龄残差实验见 [docs/FWQ_PRETRAINED_TRANSFER.md](docs/FWQ_PRETRAINED_TRANSFER.md)。

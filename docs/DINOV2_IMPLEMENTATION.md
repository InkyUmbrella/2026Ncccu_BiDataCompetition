# DINOv2接入与统一推理：工作区待验证版本

2026-10-27。用户最新安排：**将本批源码发布到GitHub，运行机器仍等待用户指定服务器或4090**。本批只做源码编辑与静态检查，没有在本机、服务器或4090运行ML测试/训练/推理；没有下载模型权重。下列命令为待执行入口，不代表已验证可用，也不代表T4性能已经达标。

## 已实现范围

- `dinov2_vits14`：原始通用、无register的ViT-S/14；本地原生PyTorch构造，SDPA注意力，保留官方state-dict键/形状及位置插值。最终LayerNorm的CLS特征→128维头→单logit；分类头dropout0.2。未加入医学模型、年龄、额外pooling或自监督训练。
- 来源固定到Meta官方原始权重直链；加载时仅读本地文件，检查SHA256、大小、LVD142M声明及固定版本的许可证/模型说明。预备脚本负责显式下载，模型构造不执行Torch Hub或自动下载。
- 训练沿用现有EMA、优化器、scaler、RNG、历史、嵌入best及逐轮不可覆盖断点；扩展初始化归档，保存原始DINOv2权重与来源材料。原ConvNeXt/历史checkpoint仍有兼容性检查待执行。
- 新增15轮DINOv2配置及10轮512备选配置。512通过`init_from_sha256`强制绑定固定旧B0快照；漏传或传错初始化会拒绝开始。resume仍只用完整last，不能同时init-from。
- `predict.py`、`cost.py`调用`glaucoma.inference`的同一加载/逐图函数。spec冻结权重、代码哈希、确定性预处理、精度、TTA及0.5阈值；加载时核对，禁止静默替换。支持单图与原图/水平翻转概率等权平均。
- 耗时按100张、额外5张预热、3轮、batch1；每次重新读图，计入解码/预处理/前向/后处理/逐行写出并flush及CUDA同步。CSV按同一`.10g`序列化后的数值和标签严格比对，不用容差隐藏差异。
- 部署大小清单统计实际加载checkpoint及spec；代码、初始化归档不被推理加载。当前只实现FP32/FP16，没有宣称支持INT8或编译引擎。

## 上游来源与本地移植

上游代码/文档版本：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`。原始权重：

https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

许可与改动归属见 [DINOv2说明](../licenses/dinov2/NOTICE.md)、[完整Apache-2.0](../licenses/dinov2/LICENSE)和[官方model card副本](../licenses/dinov2/MODEL_CARD.md)。本地实现不导入上游医学分支。它与上游的数值一致性、实际官方文件严格加载、目标形状CUDA表现仍待测试，不把静态结构相同当成已通过运行核验。

## 运行前检查顺序

运行位置、GPU及是否进入正式训练均等待用户指定。共享服务器仍只允许检查空闲后的物理GPU4/5；自有4090按实际卡号设置，设备编号不能跨机器照抄。现有共享环境保留，新环境隔离。以下为Bash示例，`/path/to/...`须按选定机器替换。

1. 先跑现有及新增合成回归，保留日志/退出码。默认不访问网络、正式图片或GPU：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=src \
  python -m unittest discover -s tests -v
```

2. 用户同意在指定机器准备权重后，设置独立缓存，显式下载官方文件和固定来源材料：

```bash
export TORCH_HOME=/path/to/task-torch-cache
python scripts/prepare_dinov2.py --cache-dir "$TORCH_HOME/hub/checkpoints"
```

脚本首次从官方无登录URL下载，记录实际SHA256；再次复制/取回时用已归档SHA验证，可使用`--expected-sha256`。不会伪造尚未下载文件的哈希。已有缓存不覆盖，失败暂存保留供诊断。后续进程需使用同一TORCH_HOME；迁移整个checkpoints内DINO文件、source.json和documents目录。

3. 在所选机器执行官方缓存加载、固定上游源码的功能对照与小规模CUDA恢复检查。这些均为工程检查，合成成绩不得当作正式AUROC：

```bash
GLAUCOMA_DINOV2_OFFICIAL=1 OMP_NUM_THREADS=2 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_dinov2.py -k original_cached -v

# 先取得并核对上述精确revision的本地上游源码；检查本身不联网。
GLAUCOMA_DINOV2_UPSTREAM=/path/to/pinned-dinov2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_dinov2.py -k matches_pinned -v

# CUDA_VISIBLE_DEVICES必须先按用户指定机器设置并核对空闲状态。
GLAUCOMA_DINOV2_SMOKE=1 OMP_NUM_THREADS=2 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_dinov2.py -k cuda_resume -v
```

小规模恢复测试使用42像素合成图片及随机初始权重；还须在正式训练前检查真实392输入、micro16/累积8的显存与FP16有限更新。T4/V100不依赖仅新架构GPU支持的自定义算子，但可运行性必须实测。4090或V100跑通不能代替T4耗时验证。

## 正式训练命令：待用户安排

主方向固定15轮、seed42、392输入、head LR1e-4/backbone1e-5、有效batch128、warmup1、EMA0.999，逐轮raw/EMA完整valid选模。仅在上述检查通过且用户指定运行后使用：

```bash
python scripts/train_audited.py \
  --config configs/experiments/dinov2_vits14_crop392_ema.json \
  --data-root /path/to/glaucoma_first --output-root /path/to/new-runs --device cuda
```

恢复时同一配置添加`--resume /path/to/last.pt`，并保留原运行目录与全部恢复链。512备选使用`configs/experiments/b0_crop512_refine_ema.json`，添加`--init-from /path/to/b0_init.pt`，要求SHA256为`c9a7c3c6cf7802b96d43bc7e89737579d43324ccd4b1afdfb76ec5fe2978bb18`；建独立512缓存或直接读原图，不复用384缓存。最多2个新配方+1次确认、40轮上限保持不变。

## 冻结预测、完整valid复核与计时

在用户指定机器、准备好推理best后执行。源码或精度/TTA变更须重新冻结；推理包保留完整项目源码结构与相对路径。spec使用LF规范化源码哈希以支持Windows/Linux迁移，权重与实际spec仍按原始字节哈希。checkpoint必须是普通推理best，不接受包含优化器的last。

```bash
python scripts/freeze_inference.py --checkpoint /path/to/best.pt \
  --output /path/to/deployment/inference_spec.json --precision fp16 --tta none

python -m glaucoma evaluate-frozen --spec /path/to/deployment/inference_spec.json \
  --data-root /path/to/glaucoma_first --split valid --output-dir /path/to/new-valid-review

python predict.py --spec /path/to/deployment/inference_spec.json \
  --data-root /path/to/glaucoma_first --output /path/to/new-submission.csv

python cost.py /path/to/new-submission.csv /path/to/pictures_100 /path/to/pictures_5 \
  --spec /path/to/deployment/inference_spec.json --output-dir /path/to/new-cost-report
```

先在完整valid重算旧refine的统一batch1基准，再比较新候选，保留历史batch32成绩。旧CLI `predict --checkpoint`仍可使用，会另外生成同名`.spec.json`并走batch1；`--batch-size`必须为1，`--num-workers`不参与逐图加载。批量`evaluate`保留，正式新候选比较用`evaluate-frozen`。工程检查使用train/valid/合成图片；test只能按规则作最终功能/推理/指定耗时用途。

`cost.py`默认核对T4及指定Python/PyTorch/torchvision/CUDA/cuDNN/驱动；其他机器只能加`--diagnostic`。CPU诊断需单独冻结FP32 spec。报告始终记录真实环境和诊断状态；CPU/内存/SSD限制另外核验，最终以主办方复测为准。年龄模型须以`--ages-csv`提供覆盖100+5图的年龄信息，不估计test统计量。

最终CSV应使用相同冻结spec在指定评测环境导出。其他GPU/框架生成的FP16结果不能未经核对就声称逐位一致；先在valid解决环境差异，不根据test结果调整模型或后处理。

## 尚未完成的验收

所有运行检查均待用户指定位置；包括现有回归、上游对照、实际权重、完整resume、392/512显存与速度、官方新软件环境和T4的100ms/500MB验收。当前没有新正式成绩，不替换现有refine。ConvNeXt具体权重许可材料缺口仍见[合规核查](COMPLIANCE_REVIEW.md)。逐样本预测留在受控运行目录；用户已授权发布本批源码，运行验收仍待安排。

# 第二轮：根据0.988527实验继续优化

本轮基于 `shimmer` 的 `3505336`，不会改写已上传的原实验记录和旧配置。代码和配置已推送，并已在授权服务器启动准备流程；CPU回归与真实图片GPU更新检查已通过。下列配置也可独立运行。当前尚无本轮完成的新AUROC成绩，整批结果将自动归档。

## 结果核查与优先级

最新上传的 `shimmer_512_pretrained_reference`：28/30轮，第23轮最佳，AUROC **0.9885271098**、AP **0.8275141457**，valid为5849图/191阳性。已核对57份归档JSON的SHA256，并确认新旧训练/验证CSV哈希一致。第8轮AUROC **0.9885104536**，和第23轮只差 **0.0000166562**；最后一轮训练loss降至0.002388，验证AUROC为0.987134。这支持先尝试较温和的微调和权重平滑，而非优先加大采样/Mixup力度；它不是关于过拟合机制的独立因果证明。

仓库只有这次实验的config/metadata/history/metrics，没有best.pt和逐样本valid_predictions.csv。因此本轮核查的是记录一致性，不能独立重算预测分数或计算与其他模型的配对置信区间。第8轮对应权重也不在仓库，不能从history还原它。详细数值见 [analysis.json](analysis.json)。

推荐顺序：

|顺序|实验配置（configs/experiments/下）|目的|
|---|---|---|
|1|`shimmer_512_reference_refine_seed42.json`|从0.988527的best.pt出发，10轮1e-5微调，比较raw/EMA，保留起点作为最佳候选|
|2|`shimmer_512_pretrained_reference_seed43.json`|完全保留成功参考配方，只换seed和实验名，验证稳定性|
|3|`shimmer_512_reference_seed43_ema.json`|和上一步相同seed及配方，加入EMA，区分权重平滑的作用|
|4|`shimmer_efficientnetv2s_384_reference_seed42.json`|增加不同骨干，评估单模型能力和融合互补性|
|按需|`shimmer_384_draft_pretrained_reference.json`|与512成功配方统一draft解码、16×8 batch及训练配方，只改变输入尺寸|

如果资源紧张，先执行1和2；若EMA确有收益，后续再投入第3项。原来的 `shimmer_512_imagenet_seed43` 含额外采样、Mixup、平滑和低骨干LR，不作为当前0.988527的首选复现对照。旧的 `shimmer_512_refine_seed43` 也是针对该融合配方的历史配置；本轮请使用名称含 `reference_refine_seed42` 的新配置。

## 1. 优先微调当前最佳模型

先拉取shimmer最新代码，在项目根目录执行（Bash示例，Windows PowerShell可将每条命令写成一行）：

```bash
export PYTHONPATH=src
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train \
  --config configs/experiments/shimmer_512_reference_refine_seed42.json \
  --data-root /实际路径/glaucoma_first \
  --init-from /原0.988527运行目录/best.pt
```

请使用最佳 `best.pt`，不要使用最后一轮的 `last.pt`。不需要下载新的ImageNet权重。数据预处理、模型头、dropout、stochastic depth、自然采样、pos_weight=5、不加Mixup/标签平滑均与原成功配方一致。

本轮微调配置：10轮，峰值LR=1e-5，warmup=1轮，骨干缩放1.0，EMA decay=0.999，EMA warmup=200次更新，每轮同时评估raw/EMA。`keep_initial_best=true` 会在训练前用当前数据和环境重评起点，并把它一并参与AUROC/PR梯形面积选模；初始checkpoint的SHA256写入metadata。若微调一直更差，best.pt保留起点权重，metrics会显示 `selected_weights="initial"`、`best_epoch=-1`（表格换算为第0轮，即尚未更新）。恢复训练继续保留这个候选。

这个保护只保证相同验证设置下的选模行为，不证明模型泛化一定改善。起点重评值若明显不同于0.988527，请先核对数据、checkpoint和运行环境。原best.pt请继续保留在原目录。

先检查入口可追加 `--stop-after-epochs 1`，之后用同一配置加 `--resume /新微调目录/last.pt` 继续。resume时不要同时传入init-from，不要改变数值配置。

## 2. 同配方seed与EMA对照

```bash
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train \
  --config configs/experiments/shimmer_512_pretrained_reference_seed43.json \
  --data-root /实际路径/glaucoma_first

CUDA_VISIBLE_DEVICES=0 python -m glaucoma train \
  --config configs/experiments/shimmer_512_reference_seed43_ema.json \
  --data-root /实际路径/glaucoma_first
```

这两组从官方ImageNet权重重新开始，不传init-from；不能用它们的配置resume原seed42实验。EMA配置新增的每轮验证不参与训练反向传播。指标同时保留在history.valid与history.valid_raw里；best.pt仍是单模型，推理不额外运行EMA分支。

## 3. EfficientNetV2-S候选

```bash
CUDA_VISIBLE_DEVICES=0 python -m glaucoma train \
  --config configs/experiments/shimmer_efficientnetv2s_384_reference_seed42.json \
  --data-root /实际路径/glaucoma_first
```

使用torchvision官方 `EfficientNet_V2_S_Weights.IMAGENET1K_V1`，固定权重版本，官方加载器验证文件哈希前缀；新二分类头为128维MLP。不添加新的推理依赖。官方定义约2150万参数；本项目替换分类头后参数量略有不同，尚未测量T4速度，不能从参数量推断100ms达标。

已核对 [torchvision 0.25.0源码](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/efficientnet.py)：该 **S** 权重使用标准ImageNet mean/std，不能误用某些其他EfficientNetV2权重的0.5归一化。保留眼底完整视野，沿用裁黑边/letterbox而非中心裁剪。恢复、迁移、评估和预测复用已有离线checkpoint机制。该骨干有BatchNorm，16×8累积不等同于128张的BatchNorm统计，不建议随意改成特别小的物理batch。

先用384控制成本；只有实际表现和T4预算支持时再扩到512。它是待验证候选，不保证高于ConvNeXt。暂不加入ConvNeXt-Small或整图+视盘双流，避免同时扩大实验范围。

## 4. 收集、比较与部署

对每个候选使用统一的官方valid，保留运行目录的config/metadata/history/metrics、best.pt和valid_predictions.csv。汇总支持flat评估指标及wrapped训练指标；新增JSON字段报告实际完成轮数、计划轮数和从1开始的最佳epoch，避免把28/30记作完成30轮：

```bash
python scripts/report_runs.py --root outputs/shimmer --json > outputs/round2_runs.json
python scripts/compare_predictions.py /候选/valid_predictions.csv /原0.988527/valid_predictions.csv --reps 1000
```

配对比较需要两份逐样本CSV，不能根据summary数字生成置信区间。现有比较脚本预设0.003的最小效应门槛，不应把它的NO-EVIDENCE标签直接解释为两个模型相同；同时查看点估计、区间和多seed稳定性。官方valid反复选模导致的偏差不会因bootstrap而消失。

先保留较好的单模型。若seed43或EfficientNet的错误模式有互补，再利用现有fuse_predictions.py测试少量预先固定的等权融合；不要在191个阳性上大规模搜索融合权重。rank融合只用于离线分析，不能冒充独立单图概率。

新模型用正常predict入口生成提交，延迟工具会读取checkpoint内的模型和归一化：

```bash
CUDA_VISIBLE_DEVICES=0 python -m glaucoma predict \
  --checkpoint /选定运行/best.pt --data-root /实际路径/glaucoma_first \
  --output outputs/submission_round2.csv

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_latency.py \
  --checkpoint /选定运行/best.pt --data-root /实际路径/glaucoma_first \
  --images 200 --device cuda:0 --output outputs/latency_round2.json
```

仍需在T4上验证总模型文件≤500MB及单图端到端≤100ms；4090D或V100不代替T4验收。若做双模型融合，须额外测量完整双模型部署链，单模型benchmark不能代替。

## 工程验证

测试日志见本目录 `cpu-tests.log`；覆盖真实torchvision EfficientNet权重注入（模拟下载）、训练/恢复/迁移/评估/预测、官方归一化、微调起点保留及flat报告轮次恢复。所有测试使用合成数据，不代表正式训练效果。其他分支的DINOv2工作本轮不合并，其文档注明尚待运行验证；后续可以作为独立候选比较。

## 已授权服务器批次

本批获准在 `heyuhang@100.68.97.6` 使用物理GPU **2/3/4**，暂无必须释放的时间；其他GPU不使用。GitHub当前没有0.988527的权重文件或Release附件，因此启动计划调整为：GPU2从官方权重复跑seed42，然后自动以其best做上述微调；GPU3跑seed43参考；GPU4跑EfficientNetV2-S。不会把旧0.97698权重当成新0.988527权重。

`scripts/run_round2_server.py` 是本批次监督入口：只允许2/3/4，每个子进程只暴露一张物理卡；开始计算前检查compute进程，若卡被占用则拒绝启动，不切换到其他卡。先校验ConvNeXt官方权重并下载校验EfficientNet官方权重，执行真实图像GPU小样本更新检查，再用本地SSD创建512/draft和384/draft缓存。需要新工作目录和至少95GiB可用空间；脚本设置私有umask，禁止覆盖现有实验。

缓存成功后启动三路训练；GPU2在参考训练后自动继续微调。每个实验完成后自动独立重评完整valid，核对5849/191、图名顺序、标签及与训练指标的一致性，记录最佳权重SHA256/大小。每个作业异常分别记录，不影响其他卡。整批完成后输出summary和固定的1000次配对比较；训练及逐样本预测保留在服务器，归档仅收集配置、元数据、history、metrics及汇总，不上传原图或权重。

本批工作目录为 `/var/tmp/heyuhang-shimmer-round2-20261027`，监控入口为其中的 `state.json`、`jobs/*/train.log`、`jobs/*/evaluate.log`；完成后看 `archive/summary.json`。精确启动状态以后续日志及仓库归档为准，不把本节计划当作已完成训练。

实际启动与GPU检查记录见 [server_runs/20261027_gpu234](server_runs/20261027_gpu234/)。训练结束后，本机收集器会将服务器归档非强制推送到同一shimmer分支；只写入该批次结果目录，遇到并发提交先rebase，不覆盖他人代码。若本机离线，服务器训练仍独立继续，记录可稍后取回。

## 02:33巡检与无损I/O修复

首次三小时巡检发现：缓存已生成，但三路进程还在审计阶段，无history和last.pt，未开始首轮有效训练。62GiB内存环境中的未压缩两套缓存约67GiB，三路读图发生磁盘等待。8张抽样的.npy读取中位数约435ms，原图draft解码/预处理约101ms；不能继续把这个缓存当成提速。

已加入**无损PNG**缓存后端（CLI `cache-images --storage png`）。它保存同一预处理后的RGB像素，不使用有损JPEG重压缩；训练/验证增强逐像素一致性测试通过，旧npy与checkpoint继续兼容。新manifest版本4记录存储格式，仍检查CSV、每张源文件签名、解码与裁边配方及缓存形状；没有绕过审计。同时将重复的源目录resolve先去重，减少NFS元数据请求。

旧三路进程均在确认无训练checkpoint后受控停止，旧记录和缓存保留；新批次使用 `/var/tmp/heyuhang-shimmer-round2-20261028`，PNG缓存放在私有 `/dev/shm/heyuhang-shimmer-round2-20261028`。启动器先抽样64张预算两套压缩缓存，并预留至少12GiB可用内存，再允许构建缓存；若空间不足则拒绝开始，不挤占其他GPU。模型配置、数据划分、归一化及种子保持不变。

回归95项（94通过，1项GPU测试跳过），检查细节见 [巡检记录](patrol_20261028_0233/)。新批次结果归档到 `server_runs/20261028_gpu234_png`，不把旧批次的受控停止记作模型训练失败或新成绩。

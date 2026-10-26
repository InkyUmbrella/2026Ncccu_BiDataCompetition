# 参赛规则合规核查（2026-10-27）

结论：**现有训练的数据来源、train/valid用途和固定模型推理路径与已核对条款一致；最终提交合规尚未闭环。** 当前主要缺口是具体预训练权重的许可材料、统一预测/计时入口、T4端到端计时证据、声明表和最终包。历史逐轮checkpoint缺失继续如实披露。

## 核查依据与范围

- 用户提供《比赛参赛规则.pdf》，6页，SHA256：`ff6aa3adbfd3d000194253055a93dbce8a140031c8432f892046a49de8136fb6`。按页1–4第1–11条及页5–6附录A逐项核对。
- 用户补充最终硬件：**NVIDIA T4 16GB ×1**。PDF规定单张端到端平均耗时上限100ms；具体驱动/CUDA/cuDNN和截止时间仍待参赛须知补齐。
- 会话此前的主办方材料补充通知还要求训练代码/依赖、seed、完整日志、初始化文件/SHA256及每轮checkpoint时间戳/哈希。PDF未重复最后一项，不据此认定补充通知失效。
- 项目核查基点：`1afe954d22c6926f9279279386b5dfde03d66fab`及已归档历史。下述结论针对记录和代码路径，不将“本地生成CSV”写成“平台已提交”。

## 逐项结论

| 检查项 | 状态 | 现有证据与边界 |
|---|---|---|
| 非医学预训练来源（第1条） | 来源条件符合，许可待闭环 | 官方ConvNeXt-Tiny `IMAGENET1K_V1`；固定原文件SHA256，来源为ImageNet，不是外部医学骨干 |
| 赛方训练数据（第2条） | 记录与代码一致 | train 51,396张；四项训练未发现外部医学数据、自监督续训或test训练路径 |
| valid使用（第3条） | 当前方式允许 | valid 5,849张/191阳性，用于选模；未并入训练。已反复选模的偏差仍须披露 |
| 数据增强（第4条） | 当前方式允许 | 仅训练启用随机翻转/旋转/颜色增强；pos_weight类别加权允许；无损train/valid缓存可保留 |
| test使用（第5条） | 已有功能推理记录，无训练回流证据 | 存在2026-10-26生成提交CSV的记录；不是“从未访问test”，也不等于已上传平台。不得利用其输出分布摘要决定阈值、先验、模型或新实验 |
| 固定参数推理（第6条） | 已核对路径符合 | `eval()`、`inference_mode()`、固定归一化、逐图确定性裁边/补边；推理不更新权重或跨样本统计量 |
| TTA/集成（第6条） | 规则允许，尚待性能与收益验证 | 可做固定、确定性的多次前向/平均；不能保留随机训练增强或做测试时训练 |
| T4计时（第7条） | 未完成 | V100训练/FP16工程测试不能代替T4端到端实测；未发现符合要求的`cost.py` |
| 推理体积（第8条） | 当前单模型有余量，最终清单待生成 | refine best为111,732,482字节；还需统计最终实际加载文件。3.02GB训练归档不直接计入推理体积 |
| 复现材料（第10条/附录A） | 部分齐备 | 已有真实配置、源码/哈希、seed、日志、预测和初始化链；尚缺完整许可证/上游说明归集、声明xlsx、最终包及对应CSV证明 |
| 补充通知的逐轮checkpoint | 历史缺口不可恢复 | 保留的完整训练断点为B0第14/30轮、年龄第20轮、refine第10轮、restart第15轮；其余不可补造 |

训练/验证记录见 [结果汇总](TRAINING_RESULTS.md)、[归档结果JSON](../training_records/20261022-23/results.json) 和 [原始文件清单](../training_records/20261022-23/original_file_inventory.json)。代码核对位置为`engine/trainer.py`、`engine/evaluation.py`、`engine/checkpoint.py`、`data/transforms.py`和`cli.py`。

当前CSV记录还表明生成时使用batch32。后续`cost.py`要求batch1，**不能未经比对就声称两者结果完全一致**。须先在valid上把统一单图路径核对好，再固定正式导出逻辑；原CSV及哈希保持可追溯，不覆盖成“同一份文件”。

## 权重与许可证核查

### 已训练的ConvNeXt-Tiny

- 精确声明：torchvision 0.20.1，`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`。
- [官方权重说明](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.convnext_tiny.html)给出ImageNet-1K及Torchvision训练配方信息；[公开权重直链](https://download.pytorch.org/models/convnext_tiny-983f1562.pth)于本次无登录HEAD检查返回200，Content-Length为114,419,221字节，与归档大小一致。该网络检查不替代原文件SHA256核验。
- 原文件SHA256：`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`，已经在历史归档中核验。
- [Torchvision v0.20.1源码许可证](https://github.com/pytorch/vision/blob/v0.20.1/LICENSE)为BSD-3-Clause；[ConvNeXt原项目](https://github.com/facebookresearch/ConvNeXt#license)声明MIT。但是**架构原项目/库代码许可证不能直接代替该具体Torchvision权重的适用许可证明**。
- [Torchvision 0.20官方模型文档](https://docs.pytorch.org/vision/0.20/models.html#general-information-on-pre-trained-weights)提醒预训练权重可能受单独条款约束。因此当前状态为“来源及哈希已核对，具体权重许可材料待完整归集确认”，不是认定违规，也不是已完成全部许可审查。

补齐项为：准确对应的许可全文及再分发依据、该权重的官方训练配方/数据说明、公开链接与文件哈希。若许可适用范围仍不明确，应取得可归档的发布方或主办方解释，不能自行把BSD/MIT当作已获确认的权重许可。

本次公开链接检查和来源状态另存 [来源核查记录](planning/pretrained_source_review.json)；其中未下载/未固定版本的候选字段明确留空，不以占位值冒充已核验SHA256。

### 后续模型允许与排除的范围

| 候选 | 结论 | 官方依据 |
|---|---|---|
| 原始通用DINOv2 `dinov2_vits14` | 可进入接入核验，不等于已经完成下载和训练 | [官方model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)列出LVD-142M与Apache-2.0；[官方README](https://github.com/facebookresearch/dinov2#license)明确覆盖原DINOv2代码和权重；还须固定精确文件/代码版本与SHA256 |
| ConvNeXt V2官方ImageNet预训练/微调权重 | 排除 | [官方README许可说明](https://github.com/facebookresearch/ConvNeXt-V2#license)对这些权重列出CC-BY-NC，命中PDF的NC禁项；不是因为ImageNet来源不合格 |
| XRay-DINO、Cell-DINO、其他医学专用骨干 | 排除 | [同仓库相关说明](https://github.com/facebookresearch/dinov2)有医学变体/独立许可；不能把原始通用DINOv2的结论推广到整个仓库 |
| 任意医学数据二次预训练/医学自监督续训、外部医学数据微调 | 排除 | PDF第1、2条；包括用赛方医学图片再做自监督续训的路线 |
| 任意NC、gated或无可核验再分发依据的权重 | 不启用 | 与是否标注“开源”、是否可下载、模型新旧无关 |

原始DINOv2训练数据的通用来源见 [官方论文数据处理章节](https://arxiv.org/html/2304.07193v2#S3)，不是医学专用预训练。通用自然图像上游已经做过自监督学习，与本项目拿医学数据继续自监督是两种不同情形。接入时只允许精确的原始通用权重，不采用同名医学派生模型。

## 后续开发硬边界

1. **数据**：探索阶段保持官方train/valid分开；test仅作提交脚本功能检查、最终推理及规定的最终耗时检查，不作为调参与实验设计证据。不改官方标签，不从test估计统计量。
2. **来源**：新增模型先建立名称、精确权重版本、公开URL、许可全文、上游数据说明、SHA256的记录。来源/许可不清楚的权重不进入正式训练和最终包。
3. **推理**：使用冻结参数、eval模式、确定性逐图预处理和固定后处理；禁止测试时更新、跨测试样本自适应和按文件名回查预存预测。
4. **性能**：T4 16GB单卡；正式口径100张、batch1串行、3轮、额外5张预热图。每次计时覆盖读图到写出预测，GPU同步；不能把读图、预处理、结果缓存或异步计算移出计时来凑分。
5. **资源**：实际加载的去重推理权重及必要配套文件≤500MB。工程上保守按500,000,000字节验收；≤450MB、≤80ms是自定余量目标，不是赛事新增条款。
6. **证据**：新运行逐轮独立checkpoint/时间/哈希、完整日志、初始化与源码归档；历史文件缺失如实说明。新实验不能冒充旧实验的复原文件。
7. **分工**：服务器独立目录开发/跑通测试，只用已确认空闲的物理GPU4/5；队友4090执行完整训练；T4完成最终硬件验证。源码/配置/文档以inkyumbrella提交，权重归档服务器并提供哈希/取回命令。
8. **提交**：材料包CSV对应真实平台历史最高分记录；valid最高分不能代替平台历史成绩。比赛队伍名不能直接用GitHub身份推定。

## 尚待明确

截止时间、T4对应软件版本/测试环境、参赛队伍名，以及此前补充通知是否仍按原要求执行。第1条“赛后提交材料”与附录A“截止时间”的表述存在差异，准备工作按截止前齐备安排，具体时点以主办方明确通知为准。未自动执行比赛提交、撤销或对外询问。

具体开发次序、实验预算和验收条件见 [后续开发计划](DEVELOPMENT_PLAN.md)。

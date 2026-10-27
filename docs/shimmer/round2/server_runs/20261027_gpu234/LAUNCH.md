# 服务器批次启动记录

- 计算范围：仅物理GPU 2/3/4；当前无必须释放的时间。
- GPU2：seed42 512参考配方，完成后自动以其best做10轮低学习率raw/EMA微调。
- GPU3：同参考配方seed43。
- GPU4：EfficientNetV2-S 384。
- 源码快照：9d86ba4，服务器 /home/heyuhang/work/shimmer-round2-20261027。
- 工作根目录：/var/tmp/heyuhang-shimmer-round2-20261027。
- 监督PID：3585251；实时状态以工作目录state.json为准。
- 已通过真实图片GPU更新检查（各16张train、16张valid）：ConvNeXt 512峰值约5140MiB，EfficientNetV2-S 384约3652MiB。它们是工程检查，不是正式AUROC结果。
- startup.json是准备阶段快照；各任务完成后生成独立完整valid重评记录、校验信息及summary.json。
- 整批完成后，本机后台收集器自动将归档提交到shimmer；checkpoint和逐样本CSV保留在服务器，不随归档上传。

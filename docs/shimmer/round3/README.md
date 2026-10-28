# 空间注意力与年龄残差对照

目标：在独立验证 AUROC 0.9902695345 的 ConvNeXt-Tiny seed42 上测试可归因的结构变化。当前没有新结构的正式训练结果。

- control：round2 的普通微调 `shimmer_512_reference_refine_seed42`。
- spatial：从相同权重初始化，增加4头空间注意力池化与轻量残差分类器。
- age：从相同权重初始化，增加已有的年龄残差分类器。

三组训练参数相同：10 epochs、LR 1e-5、effective batch128、EMA .999，同时比较 raw/EMA。新增分支输出层为零，初始化时保留原模型预测；initial-best 保护避免退化覆盖原始模型。空间分支复用现有特征图，没有额外编码器、外部数据或区域标注。注意力不保证自动定位视盘，须通过验证判断是否有效。

每个候选都用完整5849张/191阳性验证集独立重评，核对预测顺序、标签、CSV及权重hash。首次结构实验仍不能证明跨训练随机性的稳定收益。T4延迟尚未测量；增加参数量不代表满足100ms部署要求。

`check_complementarity.py` 对seed42/43进行固定等权原始logit、等权概率融合，统计成对排序错误及配对bootstrap；不搜索融合权重、不输出逐样本数据。此验证集也用于checkpoint选择，因此区间不能消除模型选择偏差，也不反映重训波动。

服务器执行：

```sh
python scripts/run_spatial_ablation.py --previous-work /var/tmp/heyuhang-shimmer-round2-20261028 --work-root /var/tmp/heyuhang-shimmer-round3-spatial-20261028
```

仅使用物理GPU3和4，各自等待卡释放，最长24小时；复用已验证的512无损缓存。GPU2上的已有微调不受影响。每个新work-root只启动一次；失败后检查state.json再决定resume，不盲目重启。

本次本地测试使用torch2.5.1+cpu/torchvision0.20.1+cpu，与服务器主版本一致。测试日志见cpu-tests.log。SSH当前在握手阶段被远端关闭，因此尚未启动正式实验；本地恢复监视器会在连接恢复后自动部署并启动。

完整回归：97项，96通过、1项CUDA检查在CPU环境跳过。新增空间分支85,829参数，FP32参数343,316字节。

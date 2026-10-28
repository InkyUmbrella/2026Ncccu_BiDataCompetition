# 下一轮：512 的训练目标对照与跨机器评估核对

## 最新结果与选择理由

上游提交0497677目前只提供round4的README文字总结：本机初始512 AUROC0.988501，512微调0.988612，640微调0.982846。没有本轮manifest、权重hash、完整history和独立重评JSON，因此这些是用户上传的结果摘要，无法进一步核实曲线及跨机器差异的原因。

本轮停止增加分辨率/新增分支，保持ConvNeXt-Tiny 512。测试目标改成：在BCE基础上，少量加入正负样本的logistic排序损失，鼓励阳性logit高于阴性。它是AUROC的代理目标，不保证提升AUROC；对困难或错标样本也可能过拟合。

## 先确认起点

同一权重的评估分数不一致时，先核对文件与数据，再分析FP32/AMP、解码和依赖版本，不能直接归因为硬件或视作真实模型提升。

```sh
python -m glaucoma evaluate --checkpoint /path/to/best.pt --data-root /path/to/glaucoma_first --device cuda:0 --num-workers 0 --precision fp32 --hash-images --output-dir /path/to/audit_fp32
python -m glaucoma evaluate --checkpoint /path/to/best.pt --data-root /path/to/glaucoma_first --device cuda:0 --num-workers 0 --precision amp --output-dir /path/to/audit_amp
```

Windows同样可以执行这些单行命令，替换本机路径。先 `git switch shimmer` / `git pull --ff-only`，并在已有训练环境执行 `python -m pip install -e .`。

新增的provenance.json记录checkpoint SHA256、CSV SHA256、实际AMP开关、batch、完整配置、torch/Pillow等依赖版本、源代码hash和TF32开关。--hash-images还生成按CSV顺序排列的原始图片内容聚合哈希，排查CSV相同而图片不同的情况；会增加一次磁盘读取。两台机器都运行相同命令才能比较图像哈希。metrics.json与逐样本predictions.csv维持原格式；预测CSV只留本机，不提交仓库。

旧服务器0.9902695345对应原始纯图像best.pt的SHA256为：

```text
b546ffa77040c879ea6ec0b25dc27cd62e402e9c07e7efeaca3bd494de3f119e
```

如果当前文件hash不同，就不是同一份权重的跨机复核。可继续用本机确认过的512权重，但必须把它作为新实验共同起点，不能混用旧分数。**权重仍不在GitHub，必须自行迁移**。

## 三组对照

|配置|训练采样|BCE pos_weight|排序损失权重|
|---|---|---:|---:|
|natural|原始自然分布|5|0|
|sampled|阳性采样比例0.25，有放回|1|0|
|rank|与sampled完全相同|1|0.1|

sampled与rank除实验名和排序损失权重外完全一致；两者比较才能归因排序目标。natural与sampled同时改变采样及类别权重，是两种训练配方的对比，不能拆开归因。

三组都为512、10epochs、head LR1e-5、backbone LR1e-6、有效batch128、EMA/raw双候选和initial-best保护。从相同checkpoint初始化，使用相同数据与预处理。排序仅在每个microbatch内计算，**不跨梯度累积窗口配对**，因此两组microbatch必须相同；有效batch相同不代表排序目标相同。

损失：mean BCE + 0.1 × mean softplus(negative_logit - positive_logit)。梯度累积时按microbatch样本数归一化；没有同时包含两类的microbatch排序项为0。history.train会记录rank_loss和rank_batches_with_pairs。排序不使用验证集标签或跨batch队列，不支持Mixup/CutMix/标签平滑，配置会拒绝这些组合。

## 新机器运行

```sh
python -u scripts/run_ranking_ablation.py --checkpoint /path/to/best.pt --data-root /path/to/glaucoma_first --output-root /path/to/new_ranking_run --device cuda:0
```

默认一张指定的卡依次跑sampled与rank两组，优先验证排序项是否有效；追加 `--recipes natural sampled rank` 可运行完整三组，默认workers=0，规避上一轮报告的Windows epoch边界spawn崩溃。**没有实现或声称修复persistent workers**：仅给现有每轮重建DataLoader传persistent_workers=True并不能跨轮复用，且还涉及随机增强/恢复一致性，暂不改变该行为。

已有兼容的512无损缓存可以追加 `--image-cache /path/to/cache512`，三组共享同一缓存。没有缓存可追加 `--cache`，每组构建一份PNG缓存，需注意磁盘空间；建议先单独cache-images建好再共享。PNG不丢失预处理后像素。

可先加 --prepare-only 生成配置（不训练，output-root仍需新目录）。正式运行另用新的output-root。默认只运行关键两组，可用 `--recipes natural sampled rank` 加入自然采样对照，也可各用新目录并行运行，GPU由执行者指定。默认batch8、accumulation16；显存允许可两组一起改batch16，脚本保持有效batch128。不要在对比中只改一组batch。

weights、数据与输入协议沿用原始512模型；没有外部数据、新增模型参数或额外推理计算。训练端只增加microbatch内的成对损失计算。若新目标没有胜过初始模型，保留初始权重。

## 结果判断与回传

根目录manifest.json有同机baseline、每组initial/best独立重评分数、相对baseline的差值及权重hash。每组evaluation/provenance.json保留完整评估记录。日志在组目录train.log/evaluate.log；配置、history、metadata、metrics和initial_validation在training子目录。

优先比较rank与sampled，再与baseline/natural比较。selected_weights=initial意味着未改善，不能把回退的分数当新目标有效。一次seed42不能证明跨重训稳定提升；有明确收益后再做第二seed验证，不先扩充多组随机试验。

中断可沿用现有train --resume流程，不能同时传--init-from；数值配置需保持一致。启动脚本不会自动接管手动resume，手动恢复后要独立evaluate。

请回传manifest.json、三组history/metadata/metrics/initial_validation及独立重评metrics/provenance。无需上传数据、模型、逐样本预测。

## 本地验证

完整回归102项：101通过，1项CUDA检查在CPU环境跳过。新增排序测试3项通过，覆盖梯度方向、常数平移不变性、无配对batch、配置禁用混合标签、排序项实际参与训练、精确断点恢复及带图像哈希的FP32重评记录。使用torch2.5.1+cpu / torchvision0.20.1+cpu；尚未在新机器执行正式GPU训练，不能声称评分提升。

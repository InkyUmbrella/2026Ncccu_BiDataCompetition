# 无损PNG内存缓存批次

- 源码快照：40e81e9；监督PID：3611913。
- 工作目录：/var/tmp/heyuhang-shimmer-round2-20261028。
- 私有缓存：/dev/shm/heyuhang-shimmer-round2-20261028。
- 三路分配不变：GPU2 seed42后接微调、GPU3 seed43、GPU4 EfficientNetV2-S。
- 64张容量抽样估计两套PNG共约20.7GiB，加上余量需约26.4GiB；启动时MemAvailable约41.7GiB，并预留12GiB。两套缓存并行构建。
- 真实生成PNG已检查16张，像素与原预处理完全一致，内存读取中位数约8.95ms；旧npy在并行审计负载下约435ms。该抽样不是端到端训练加速比。
- 02:33巡检时授权卡已被lichunyu的其他进程占用。每个任务独立等待对应物理卡释放，最多等待24小时，不中断他人进程，不切换到其他卡。
- 训练结束后自动完整重评、核验并汇总；后台收集器只向本目录归档JSON和日志，权重/逐样本预测留在服务器。
- startup.json是准备阶段快照。实时状态请查看服务器state.json，完成后查看summary.json。

# GPU 0–5 共同计算与通信可行性报告

**结论：六卡 DDP 的基础通信和短时稳定性有条件可用，已验证的设置是 `NCCL_P2P_DISABLE=1`。默认 P2P 配置不可用。**

已完成小型 DDP 梯度同步、不同消息大小的 FP32/FP16 通信，以及 120 秒持续通信。尚未测量比赛模型的训练速度、长时间稳定性、数据读取或模型收敛；当前没有编写或运行比赛训练程序。

**环境与边界**

| 项目 | 实测情况 |
|---|---|
| GPU | 0–5：Tesla V100-SXM2-16GB，每卡 16 GiB |
| GPU 拓扑 | 0–3 在 NUMA 0，4–5 在 NUMA 1；跨节点为 SYS |
| NVLink | 用户与管理员沟通后确认：服务器未安装 NVLink；与拓扑查询及 inactive 结果一致 |
| CPU | 双路 Xeon E5-2680 v4，共 56 逻辑 CPU |
| 驱动 | 570.211.01 |
| Python / PyTorch | Python 3.11 / 2.5.1+cu121 |
| CUDA runtime / NCCL | 12.1 / 2.21.5 |
| 共享内存 | `/dev/shm` 32 GiB，检查时基本空闲 |
| 项目存储 | `/home` 为 NFS；本次不测数据加载性能 |

六张 16 GiB 显卡在普通 DDP 下各保存一份模型和各自批次，不构成可供单卡直接使用的 96 GiB 显存。

管理员确认未安装 NVLink，禁用 P2P 后，NCCL 可通过主机共享内存中转。本次日志明确显示 `SHM/direct/direct`，且六卡 DDP 梯度同步与 19,940 次持续通信校验通过。当前结论是：NVLink 不可用，默认 P2P 路径在测试中失败，主机内存中转的六卡通信可用。是否采用多卡训练，还需结合后续实际模型吞吐衡量。

测试仅使用 GPU 0–5。GPU 6–7 的既有进程未被修改；检查前后其 PID 和显存用量一致。没有改动驱动、内核、系统 NCCL 配置或 `.bashrc`。

**实验结果**

| 配置 / 测试 | 结果 |
|---|---|
| 六卡，默认 P2P | 首次 4096 元素 ALLREDUCE 在 30 秒后超时；进程退出码 1 |
| 六卡，仅关闭 cuMem：`NCCL_CUMEM_ENABLE=0` | 使用 P2P/IPC 仍在首次 ALLREDUCE 超时；退出码 1 |
| 六卡，仅禁用 P2P | 六个 rank 的 CUDA 计算、ALLREDUCE、DDP 构造和梯度平均全部通过；退出码 0 |
| 六卡，禁用 P2P，消息大小与持续通信测试 | 所有规模、类型及六个 rank 均通过；退出码 0 | 
| GPU 0、1，默认 P2P | 同 NUMA 的双卡仍在首次 ALLREDUCE 超时；退出码 1 | 
| GPU 0、1，禁用 P2P，64 MiB | FP32/FP16 求和与广播全部通过；退出码 0 | 
| GPU 0、4，禁用 P2P，64 MiB | 跨 NUMA 的 FP32/FP16 求和与广播全部通过；退出码 0 | 

失败实验均设置 30 秒进程组超时及 100 秒外层超时，并禁用 core dump。成功的六卡探针使用 `SHM/direct/direct` 通道。

这些对照支持故障与 P2P 通信路径有关；仅切换 cuMem 为 IPC 不够。GPU 0、1 的失败表明故障不局限于跨 NUMA 连接。尚不能凭这些结果认定某块 GPU 损坏，或确定 ACS、IOMMU、驱动、NCCL 中的某一项就是根因。

**正确性与持续测试**

- 测试类型：FP32、FP16；操作：ALLREDUCE 求和、轮换源 rank 的广播。
- 每卡消息大小：4 KiB、1 MiB、16 MiB、64 MiB、256 MiB。
- 每种组合先预热 5 次，再测量 30 次；全部结果逐元素校验。
- 求和输入随 rank 和迭代变化；广播源随迭代轮换，避免只验证固定常数路径。
- 持续阶段：16 MiB FP32 消息，实际 120.037 秒，共 **19,940 次**载荷求和/广播，每次在所有 rank 校验结果，无错误或超时。
- 持续循环由 rank 0 广播退出条件，所有进程执行相同数量的操作。

这是短时稳定性证据，不是多小时训练的稳定性保证。

**通信耗时**

六卡禁用 P2P 后，FP32 ALLREDUCE 的结果：

| 每卡消息大小 | 平均耗时 | 有效载荷吞吐 |
|---|---:|---:|
| 4 KiB | 0.133 ms | 0.031 GB/s |
| 1 MiB | 0.502 ms | 2.089 GB/s |
| 16 MiB | 7.888 ms | 2.127 GB/s |
| 64 MiB | 31.586 ms | 2.125 GB/s |
| 256 MiB | 126.420 ms | 2.123 GB/s |

FP16 同字节量的结果相近：64 MiB 31.748 ms、256 MiB 127.534 ms。此表按字节量比较，不表示同数量 FP16 和 FP32 元素需要相同时间。

64 MiB FP32 ALLREDUCE 对照：

| GPU 配置（均禁用 P2P） | 平均耗时 | 有效载荷吞吐 |
|---|---:|---:|
| 0、1，同 NUMA | 16.712 ms | 4.016 GB/s |
| 0、4，跨 NUMA | 10.478 ms | 6.405 GB/s |
| 0–5，六卡 | 31.586 ms | 2.125 GB/s |

本次双卡样本中跨 NUMA 反而更快，不能简单按拓扑标签推断速度；未进行 CPU 绑核或内存局部性调优。这些结果记录当前配置的表现，不能单独归因于 NUMA。

计时从主机发起操作到 GPU 完成，排除输入填充、结果校验和预热；每次取最慢 rank 的耗时，再对 30 次取平均。GB/s 为十进制载荷字节数除以耗时，不是实际 PCIe 总线带宽，也不是训练吞吐。通信时间与模型计算可部分重叠，因此不能直接推导实际 epoch 时间或六卡加速比。

**可复现命令**

在服务器 Bash 中运行最小 DDP 探针：

```bash
cd ~/competition
conda activate glaucoma
timeout -k 10s 100s env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node=6 \
  scripts/check_ddp.py --timeout 30
```

重跑通信规模及持续测试（每次创建新目录，避免覆盖原证据）：

```bash
cd ~/competition
conda activate glaucoma
result_dir=$(mktemp -d "$PWD/outputs/comm-check-XXXXXXXX")
set -o pipefail
timeout -k 10s 360s env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 \
  NCCL_DEBUG=WARN OMP_NUM_THREADS=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node=6 \
  scripts/check_collectives.py --iterations 30 --seconds 120 \
  --output "$result_dir/results.json" 2>&1 | tee "$result_dir/run.log"
```

六个 rank 均显示 `ALL CHECKS PASSED` 且退出码为 0 才算通过。环境变量只对本次命令生效，无需永久写入系统配置。

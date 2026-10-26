# 2026-10-22/23 历史训练材料

结果与证据缺口请先阅读 [正式结果报告](../../docs/TRAINING_RESULTS.md)。本目录原始材料于2026-10-26从服务器收集，原文件字节保持不变；`.gitattributes`对本目录关闭文本行尾转换，以便跨Windows/Linux重算SHA256。

- `runs/b0`、`runs/age`、`runs/refine`、`runs/restart`：各项最终实际config/metadata/history/metrics、age_stats、完整valid预测及data_audit。
- `runs/b0_before_migration`：B0迁移前的配置、metadata和历史，不能单独当作完成30轮。
- `support/`：原日志、exit、launch、context、迁移/监督/验证脚本和数据缓存manifest；编号仅用于避免重名，原路径见`original_file_inventory.json`。
- `sources/b0.tar.gz`、`age.tar.gz`、`improve.tar.gz`：分别收集的历史项目代码；内容哈希与训练时记录的对应关系见`source_provenance.json`。
- `results.json`：四项完整正式结果、最佳epoch/variant、指标/差值/配对区间、初始化/best/last/日志/预测及哈希。
- `retained_checkpoint_objects.json`：10个去重权重文件的索引，字段`file`相对下面的服务器归档根目录。
- `original_file_inventory.json`：128项原始文件的路径、当前观测mtime、字节数及SHA256；时间戳不是第三方时间认证。
- `manifest.json`：收集器生成的材料逐文件SHA256；后来编写的本README、归档下载信息等不冒充原始材料。
- `collection.log`、`archive_manifest.json`、`verification.json`：本次收集过程、服务器归档包的哈希和CPU核验结果（119个记录文件、10个权重、篡改拒绝检查）。

## 完整二进制归档

服务器上独立且不覆盖旧实验的归档目录：

```text
/home/lichunyu/competition-training-handoff/history_20261026T145022Z
```

其中`records.tar.gz`为原始文字/预测/源码材料，`weights.tar`包含10个现存二进制文件，按SHA256命名并去重；不是75轮逐轮checkpoint全集。原实验目录中的文件也仍保留。

在队友电脑下载并解包（普通终端，当前目录建议预留至少7GB空闲空间用于归档和展开的权重）：

```text
scp lichunyu@100.68.97.6:/home/lichunyu/competition-training-handoff/history_20261026T145022Z/weights.tar .
scp lichunyu@100.68.97.6:/home/lichunyu/competition-training-handoff/history_20261026T145022Z/archive_manifest.json .
tar -xf weights.tar
python scripts/verify_training_records.py --records training_records/20261022-23 --weights-dir weights
```

SSH使用已有账号授权。不要把密码或私钥加入Git。本仓库不提供其他用户的服务器认证权限。

`weights.tar`大小3,019,847,680字节，SHA256：

```text
8bc7745e1f8bcddaf83ccd476be6951a967a7cb88f63761bf08bd0836b6b6d22
```

只验证Git中的原始材料，不下载权重：

```text
python scripts/verify_training_records.py --records training_records/20261022-23
```

省略`--weights-dir`时输出`weights_checked=false`，不能把它当作已核对二进制权重。

## 只取当前最佳权重

refine第1轮raw best：

```text
scp lichunyu@100.68.97.6:/home/lichunyu/competition-training-handoff/history_20261026T145022Z/weights/1c8875a26260dd41c527d059d1421d2cefb6e5b619550ac2d0888052803f038a.pt refine_best.pt
```

文件SHA256即文件名去掉`.pt`，可用`Get-FileHash refine_best.pt -Algorithm SHA256`（PowerShell）或`sha256sum refine_best.pt`（Linux）核对。该文件为可直接加载的普通推理checkpoint。若继续某个未完成训练，则使用对应完整`last.pt`，不能以这个best冒充完整恢复断点。

用于新一轮B0后续微调的原B0固定best也在归档中：`weights/c9a7c3c6cf7802b96d43bc7e89737579d43324ccd4b1afdfb76ec5fe2978bb18.pt`。用`--init-from`会开始独立实验并重置优化器/历史；不会补回旧实验缺失的逐轮文件。

## 复核方式

`scripts/archive_training_history.py`是在服务器运行的只读收集/核验脚本，运行时`CUDA_VISIBLE_DEVICES`必须为空；`--output`必须是新目录，拒绝覆盖，且不得位于旧实验目录内。它只核对现存文件、完整valid预测和已有历史，不执行推理或训练；验证/归档日志与归档清单保存在同一服务器归档旁。

训练源码归档中可能保留旧脚本的原始服务器路径/GPU启动参数，仅用于历史溯源；后续训练使用 [4090交接文档](../../docs/TRAINING_4090.md) 的通用入口。

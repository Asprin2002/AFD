# 新版框架内的 AFD 策略验证（2026-10-11）

本页结果使用修复候选筛选前的动态策略。现行策略标记为
`phase-screen-v2`，已取消固定前三组限制；本页数值尚未用现行策略重跑，
不能直接作为新策略的效果结论。

## 比较对象

只使用 `layout_aware` / `layout-v1` 后端。对每个完全相同的工作负载和成本参数，
比较固定 AF 布局 `layout_fixed` 与 AFD 动态策略 `layout_greedy`。
吞吐收益定义为 `AFD 吞吐 / 固定 AF 吞吐 - 1`。两者均从 AF 布局开始；
固定策略不改变布局，动态策略可准备权重、复制已有 KV、改变未来 KV 写入侧和算子区间。

## 可复现命令

```bash
python3 -m unittest tests.test_layout_aware
python3 -m tools.run_layout_afd_validation \
  --config configs/layout-aware-shared.yaml \
  --attention-bandwidth-scales 1,0.5 \
  --output results/layout-afd-validation-v1/paired.csv
python3 -m tools.run_layout_afd_validation \
  --config configs/layout-aware-shared.yaml \
  --input-lengths 100000 --shared-fractions 0.5 \
  --attention-bandwidth-scales 0.6,0.75,0.9 \
  --output results/layout-afd-validation-v1/bandwidth-threshold.csv
python3 -m elasticafd simulate configs/layout-aware-online-validation.yaml \
  --policy layout_fixed --output-dir results/layout-afd-validation-v1/online-default-fixed
python3 -m elasticafd simulate configs/layout-aware-online-validation.yaml \
  --policy layout_greedy --output-dir results/layout-afd-validation-v1/online-default-afd
python3 -m elasticafd simulate configs/layout-aware-online-validation-slow-attention.yaml \
  --policy layout_fixed --output-dir results/layout-afd-validation-v1/online-slow-fixed
python3 -m elasticafd simulate configs/layout-aware-online-validation-slow-attention.yaml \
  --policy layout_greedy --output-dir results/layout-afd-validation-v1/online-slow-afd
```

脚本生成闭批、4 请求、每请求输出 32 token 的受控场景。
输入长度为 4K、32K、100K，显式共享前缀为输入的 0% 或 50%。
共享由前缀 ID 声明；相同长度不自动共享。每一对实验仅改变策略。
所有 15 对运行均正常完成，固定与动态策略生成相同 token 数。
专项测试 20 项通过。

## 结果

| Attention 有效带宽系数 | 输入 / 共享前缀 | 固定 AF 吞吐 | AFD 吞吐 | AFD 收益 | AFD 动作 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 | 4K / 0 或 2K | 38.550 | 38.550 | 0% | 0 |
| 1.0 | 32K / 0 或 16K | 38.550 | 38.550 | 0% | 0 |
| 1.0 | 100K / 0 或 50K | 24.443 | 24.443 | 0% | 0 |
| 0.5 | 4K / 0 或 2K | 38.550 | 38.550 | 0% | 0 |
| 0.5 | 32K / 0 或 16K | 38.549 | 38.549 | 0% | 0 |
| 0.5 | 100K / 0 | 12.951 | 12.951 | 0% | 0 |
| 0.5 | 100K / 50K | 12.951 | 17.317 | +33.71% | 2 |
| 0.6、0.75、0.9 | 100K / 50K | 各自成对比较 | 与固定 AF 相同 | 0% | 0 |

在唯一有收益的场景中，平均 TPOT 为固定 AF 的 308.86 ms、AFD 的
230.99 ms；AFD 执行一次共享历史 KV 复制和一次算子区间调整。
准备过程暴露时间为 0.983 s，GPU 含预留峰值为 219.73 GiB，
PIM KV/权重含预留峰值为 1099.20 GiB，均在当前模型容量内。

另用一个错峰到达的小型在线 trace 检查重规划：两个 100K 长请求共享
50K 前缀，4K 和 32K 请求分别在 0.5 s 和 1 s 后到达；长请求输出 32 token，
短请求输出 16 token。默认带宽下，固定 AF 与 AFD 均为 28.578 token/s，
AFD 无动作；带宽系数 0.5 时均为 17.787 token/s，AFD 仍无动作。
两对运行均完成全部 96 个 token。该场景说明，即使 PIM 阶段成为瓶颈，
当前候选动作和短窗口成本判断也未必选择迁移。

## 当前判断与边界

在现有默认成本参数及上述受控负载下，**尚未观察到 AFD 策略收益**。
模型能在 GPU/PIM Attention 有效带宽同时降低、且有大规模共享历史的组合下
展示条件性收益，
但系数从 0.5 增至 0.6 时该收益消失，说明结果对成本假设敏感。
因此不能把 +33.71% 表述为真实硬件收益或一般性的 AFD 加速。

这里的 100K prompt、50K 共享前缀是合成压力负载；闭批请求的 p95 TPOT
与平均 TPOT 相同，在线测试仅有 4 个请求，不能验证真实尾延迟。
模拟未包含 prefill 干扰。
动态策略的 Python 搜索墙钟时间单独报告且未计入模拟延迟；有收益场景为
2.13 s。进一步判断需用真实长度/到达/共享 trace、实测 PIM 算子及传输参数，
再重复同后端成对实验，并检查搜索时间能否满足在线决策预算。

# ElasticAFD 项目交接 · 给新一轮对话的完整上下文

> 生成日期：2026-10-11。直接粘贴本文件内容给新会话即可开工。

## 你在哪个项目里

工作目录 `/Users/arasaka/Documents/ict_code/AFD`(**不是 git 仓库**)。这是一个 **GPU + DIMM-PIM 异构 LLM 推理调度**的模拟器项目,研究问题是:

> **把解码(decode)阶段里"Attention 放 PIM、稠密算子(QKV/O投影/FFN)放 GPU"的默认分工打破——允许未来-KV 写入侧切换和算子迁移——能否缩短 CHIME 式双子批交叠流水线里较长的那一阶段。目标是找出有效与无效的工作负载区域,而不是假定迁移总有益。**

**硬性工作约定**:①所有回复用中文,新术语第一次出现必须用一句话解释,不许甩项目黑话;②把用户当严苛的论文审稿人,不预设了解领域背景(电子学、内存、LLM 推理都要从零讲清);③用户讲的不对要直说,不要顺着;④数字只有在**真实跑过**之后才能写进 CURRENT_STATE / docs,机制预测只能标成预测;⑤实验默认自动批准。

## 规范模型(唯一口径)

- GPT-175B,96 层,FP16,TP=8,DP=1,8×A100(640 GiB HBM、16.3 TB/s)。
- DIMM-PIM:2048 GiB、13 TB/s 聚合,4 个 rankset,默认一层一个 rankset;共享 60 GB/s 半双工 GPU↔PIM 链路。
- 到达语义 = prefill 已完成、decode 就绪;prompt KV 初始在 PIM。
- 硬件参数在 `elasticafd/specs.py`(preset `chime_a100`;另有 `workstation_pim_*` 单卡工作站剖面),来源追踪在 `PARAMETER_PROVENANCE.md`。
- 三种子批组织:`.lpt`(已按用户决定退役,不做对照臂)、`.alignment`(对齐式=仿 CHIME,先在 LPT 初始分组上做保持人数/每卡副本均衡的交换搜索,最小化两阶段流水不匹配)、`.fixed`(CHIME 基线自带 N=1 配对)。

## 已被观测的可信结果(全都跑过,不是预测)

1. **静态 Attention-only 网格**:batch{1..32}×input{128..8192}×output{256..8192} 共 96 形状,95 个容量可行。Top:batch16/input5200/output8192→1.5074x、batch32/2048/4096→1.4847x、batch32/2048/8192→1.4001x、batch16/5200/4096→1.3742x、batch8/8192/8192→1.2694x。收益集中在**大批量+长上下文+长输出**;batch 2/4 无正收益;batch1 最佳 1.0739x。
2. **静态稠密迁移扫描**(条件能力,未标定):最佳 B2/i128/o256 用 **PIM QKV+全 O投影**(1.4295 warm/1.2949 cold);**最大收益来自 QKV/O投影,不是 FFN**。
3. **在线原生 512 过载 trace**(qwen-coder):CHIME 基线与 joint 1.098x、吞吐+9.8%、P95−8.4%。**注意:当时 joint 用 LPT、基线用 CHIME 配对,归因混淆;且 LPT 已退役**。
4. **在线低负载稳定 trace(对齐式组织,唯一正式在线三臂)**:用 `tools/rescale_online_load.py` 把 512 trace 时间轴放大 58.3 倍降到 ≈77 tok/s(0.7×容量),得 `qwen-coder-stable-512.csv`。三臂结果(`results/full-joint-qwen-coder-stable-alignment/`):

| 臂 | 吞吐 | E2E均值 | E2E p95 | 平均TPOT |
| --- | ---: | ---: | ---: | ---: |
| chime_baseline | 72.18 | 84.75s | 261.1s | 107.4ms |
| control(对齐式) | 72.18 | 84.65s | 260.8s | 107.3ms |
| joint(对齐式) | 72.53 | 83.69s | 256.8s | 106.7ms |

joint vs control **只 +1.15%均值/+1.51% p95/+0.49%吞吐**,代价 **976 次 KV 切换 + 358 次算子切换**。机制:平均并发 7.6 时 GPU 路径利用率已 0.97,收益公式的"GPU 空闲容量"乘子≈0。**低负载甜区薄收益;大批量(batch≥16)的收益在更高并发/长输出区,单副本设备服务不到的区间**。这是当前最诚实的结论。
5. **M0–M4 静态异质联合矩阵**(`elasticafd/segmented.py` 管线):5 个锚点格子里 **joint==attention(收益全是 KV 落点,算子计划贡献为 0)、free 搜索在真实价格下重估==真实搜索(经济学轴对大批量不敏感),batch16/5200/8192 复现 1.503x(闭包 `_kv_closure` 把它从不可行救回来),batch8/8192/8192 记录为诚实败例(+18% 输给自己 best_static)**。
6. 最近一次分解(未跑新模式):在线 joint.json 的决策记录显示 527 个决策里约 90% 有算子放 PIM、主力是 O投影(部分比例),QKV 只有 2 个时刻(与 KV 迁移 XOR 互斥),FFN 15 个时刻。**在线确实触及算子迁移,但该负载(长上下文 5–10k、中批量)正好落在算子迁移的无效区(短上下文+极低批量才有效)**。

## 代码库地图

- **core**:`elasticafd/cost.py`(roofline+96 层事件 DAG+子批划分)、`simulator.py`(迭代级事件模拟)、`dense_offload.py`(算子迁移菜单 DensePlan + 权重拷贝)、`segmented.py`(静态矩阵管线:DCSchedule/state_views/`_kv_closure`)、`joint.py`/`policy.py`(策略族)、`state.py`(RequestState:只有 segments+tail_location,**没有块表**)、`config.py`/`specs.py`、`chime_baseline.py`(独立 CHIME 风格基线,非官方复现)、`capacity.py`、`pipeline.py`。
- **runners**(tools/):`run_full_joint_trace.py`(四模式 control/attention/f_all/joint)、`run_full_joint_512.py`(在线多臂,带配置哈希校验+token 守恒)、`rescale_online_load.py`、`run_heterogeneous_joint_matrix.py`、`prepare_static_recipes.py`、`prepare_burstgpt_trace.py`。
- **traces**:`workloads/static/`(同质锚点+异质剖面)、`workloads/public/`(BurstGPT 窗口、qwen-coder 系列、mooncake/qwen-thinking 候选)。
- **结果**:`results/full-joint-qwen-coder-stable-alignment/`(三臂 checkpoint json)、`results/heterogeneous-joint-matrix-recheck{,-extra,-freeb32}/`(矩阵 csv+schedules)。
- **文档**:`CURRENT_STATE.md`(当前状态权威记录)、`docs/全周期实验汇总.md`、`docs/在线低负载对齐式组织实验.md`、`docs/异质联合静态矩阵实验_飞书汇报.md`、根目录各 RUN 报告、`PARAMETER_PROVENANCE.md`。
- **测试**:168 个测试函数/20 文件,含 `_kv_closure` 回归。

## 重要:设计文档 vs 代码的差距(2026-10-11 精读分析结论)

根目录 **`Attention动态调度建模与设计收敛.pdf`(10 页)** 是成体系的**蓝图**,不是实验结果——它自己写明"硬件成本尚待校准、完整策略尚未接入模拟器"。核心内容:①两阶段联合目标式(1)(2);②每个请求一个连续 PIM 区间 + 显式 Merge 算子;③成本模型:M_p 共享/私有 KV 聚合、roofline 形状敏感、每执行组带宽、边界通信打包;④运行时状态四张表(请求/KV 块表/引用关系/权重准备)+ SGLang RadixAttention 式前缀树;**⑤动态调整:先改未来 KV 写入侧(无历史拷贝、渐进),再历史 KV 复制/算子区间扩展,短窗口式(10)比较、准备去重**;⑥贪心搜索框架;⑦稀疏模式与下一批工作。

**精读后的诚实差距结论**(这一条最值得新对话接着做):
- 时序轴:代码做了**96 层事件 DAG 调度器**,比文档建议的"形状→延迟查表"**还细**(文档§3.4 明确说不要构造在线事件模拟器)——这条可能过头。
- 数据布局轴:**结构性缺失**——①共享 KV/M_p 聚合不存在(`pim_layer_cost` 只是机械求和,M_p≡1,没有前缀共享,10 个请求共享前缀照样付 10 倍);②KV 不寻址到 DIMM/rank,局部瓶颈建不出来(带宽是固定除数);③文档 §4 的四张表在 `RequestState` 里几乎不存在(无块表/引用计数/准备状态)。
- 在线策略是两点梯形积分近似式(10),权重拷贝成本只上报不计入模拟时延。
- 未标定大头:`dense_bandwidth_scale=1.0`、`bubble_free_pipeline=True`、PIM softmax 不收费、launch=0、`exposed_communication_fraction=1/4`、FFN GELU 每元素 16 算子的假设。

## 挂起的下一步(按文档和自己的清单)

1. 在**同一条 stable 512 trace、同一 code hash**上补跑 `attention`(纯 KV 迁移)和 `f_all`(纯算子迁移),完成四模式归因(与现有 chime_baseline、joint 对齐;跑完才能把"算子=0"从推断变观测)。
2. `attention`/`f_all` 与 joint 用**同一对齐式子批组织**,排除组织方式混杂。
3. 用更长驻留/更大规划视界 + 源瓶颈保护重查 976/358 次高频切换,在别的 trace 上验证。
4. 对未标定参数做敏感度扫描(PIM 稠密/merge 成本、权重拷贝策略、链路带宽、段开销),再谈"稠密迁移"是否成立。
5. 按文档方向:**给事件模型补第一个"逻辑块 + 引用计数"模型(暂不做物理寻址)**,或反过来把事件模型降级成"形状→延迟"接口以省算力——这两条路线哪个优先值得用户自己拿主意。

## 新对话如果只有一句话,这个方向最该问

> "用现有 `rankset_event` 模拟器把文档 §4 的 KV 块表 + M_p 共享聚合补上(至少一个最小版),评估它对 1.5x 那批锚点数字和在线 512 结论的影响——这是文档与代码差距里承重墙级的那一块。"
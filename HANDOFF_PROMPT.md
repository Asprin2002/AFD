# ElasticAFD 项目交接 · 给新一轮对话的完整上下文

> 生成日期：2026-10-11；修订：2026-10-14。直接粘贴本文件内容给新会话即可开工。

> 2026-10-17 第四阶段：共用请求变化记录、同一候选各场景的相同执行部分，并在收益上限严格无法获胜时提前停止。默认 `bounded`，保留 `reference` 完整重放与 `shared` 完整得分对照方式。原 39 次决策与模型输出一致，62 个淘汰上限经完整参考得分核对。32 活跃请求的正式单次对照为 114.157／104.076／101.567 秒（完整／共用／再提前停止），动作与收益一致，11 个淘汰上限核对通过。全量 267 项测试通过。大负载仍远超在线时限；不能继续声称所有候选都有完整分数。最新先读 `docs/搜索耗时第四阶段.md`。

> 2026-10-16 第三阶段历史记录：继续精简传输排程与跨层公共输入检查。三个原窗口全部 39 次搜索、候选得分与模拟结果保持一致；关闭分项计时的四次抽查约再快 1.23–1.28 倍。32 个活跃请求／H128／12 候选／4 场景的单次压力对照由 154.902 降至 114.326 秒，评分与动作完全相同，仍远超在线时限；64 请求在接纳 54 个后因容量不足被拒绝，两版一致。全量 250 项测试通过。CPU 搜索耗时仍未计入模拟设备时间。该阶段记录见 `docs/搜索耗时第三阶段.md`。
> 最新进展：`layout_aware` 为 `layout-v4-link-calendar`；前台边界与后台准备传输共用链路日历，阶段瓶颈按逐层关键路径归因。全量回归 214 项通过，随后新增抢占边界用例并通过布局专项 47 项测试。B2/I128/O4 成对 smoke 已完成；v4 尚无完整形状矩阵或策略收益结论。下文早期 v1/v2 结果均为对应版本的历史记录，详情以 `CURRENT_STATE.md` 和 `CHANGELOG.md` 为准。

> 2026-10-16 第二阶段历史记录：复用输入相同的运算和单层结果，逐层执行与传输仍完整推进。相对第一阶段，busy／Mooncake／Thinking 分别再快约 2.80／3.35／2.60 倍，39 次决策及全部候选结果完全一致。242 项测试通过。单次平均仍约 0.4／3.1／9.7 秒，尚不满足在线时限。该阶段记录见 `docs/搜索耗时第二阶段.md`。

> 第一阶段历史记录：两个原在线窗口约快 2.20／2.31 倍，30 次决策及模型结果完全相同；未减少候选、场景或预测长度。详见 `docs/搜索耗时第一阶段.md`。原在线实验交接在主目录的 `results/online_trace_2026-10-16/SEARCH_TIME_HANDOFF.md`，不在 Git 中。

## 你在哪个项目里

工作目录 `/Users/arasaka/Documents/ict_code/AFD`（Git 仓库，远端 `git@github.com:Asprin2002/AFD.git`；`results/` 不纳入版本控制）。这是一个 **GPU + DIMM-PIM 异构 LLM 推理调度**的模拟器项目,研究问题是:

> **把解码(decode)阶段里"Attention 放 PIM、稠密算子(QKV/O投影/FFN)放 GPU"的默认分工打破——允许未来-KV 写入侧切换和算子迁移——能否缩短 CHIME 式双子批交叠流水线里较长的那一阶段。目标是找出有效与无效的工作负载区域,而不是假定迁移总有益。**

**硬性工作约定**:①所有回复用中文,新术语第一次出现必须用一句话解释,不许甩项目黑话;②把用户当严苛的论文审稿人,不预设了解领域背景(电子学、内存、LLM 推理都要从零讲清);③用户讲的不对要直说,不要顺着;④数字只有在**真实跑过**之后才能写进 CURRENT_STATE / docs,机制预测只能标成预测;⑤实验默认自动批准。

**用户表达偏好（2026-10-16 明确确认）**：先直接说要做什么，再说具体记录什么、修改什么、为什么；使用日常语言和短段落，减少专业名词。汇报时说明已完成的结果和仍存在的问题，不堆叠术语或只列抽象方案。

## 双后端现状(读代码与结果前必读)

- **`legacy` 后端**(旧,默认,`cost_backend: legacy`):`simulator.py`+`cost.py`+`segmented.py`+`dense_offload.py`+`joint.py`。用法是逐 token 事件模拟 + 96 层事件 DAG。**无块表 / 无共享 KV / 无历史拷贝**。
- **`layout_aware` 后端**(新,opt-in,`cost_backend: layout_aware`):`elasticafd/layout_aware.py` + `tests/test_layout_aware.py`(34 项,2026-10-12 实测)+ `tools/run_layout_*.py` + `configs/layout-aware-*.yaml`。按 `Attention动态调度建模与设计收敛.pdf` 实现显式共享 KV 前缀、物理副本与引用、连续 PIM 区间、Merge、历史复制 / 写侧切换 / 算子调整、逐层两阶段目标和有界候选。当前版本 `layout-v2` / `phase-screen-v3`；参数**未标定**,一律视为机制估计。
- **两后端不可混比**:`LAYOUT_AWARE_MODEL.md` 明确警告"不要用新模型延迟除以旧模型延迟当策略加速"。比较策略必须同后端成对。`run_simulation` 按 `cost_backend` 分派。

## 规范模型(唯一口径)

- GPT-175B,96 层,FP16,TP=8,DP=1,8×A100(640 GiB HBM、16.3 TB/s)。
- DIMM-PIM:2048 GiB、13 TB/s 聚合,4 个 rankset,默认一层一个 rankset;共享 60 GB/s 半双工 GPU↔PIM 链路。
- 到达语义 = prefill 已完成、decode 就绪;prompt KV 初始在 PIM。
- 硬件参数在 `elasticafd/specs.py`(preset `chime_a100`;另有 `workstation_pim_*` 单卡工作站剖面),来源追踪在 `PARAMETER_PROVENANCE.md`。
- 子批组织(legacy):`.lpt`(已退役)、`.alignment`(对齐式=仿 CHIME)、`.fixed`(CHIME N=1 配对)。

## 已被观测的可信结果

### legacy(旧模型,全跑过,历史可信)

1. **静态 Attention-only 网格**:batch{1..32}×input{128..8192}×output{256..8192} 共 96 形状,95 个容量可行。Top:batch16/input5200/output8192→1.5074x、batch32/2048/4096→1.4847x、batch32/2048/8192→1.4001x、batch16/5200/4096→1.3742x、batch8/8192/8192→1.2694x。收益集中在**大批量+长上下文+长输出**;batch 2/4 无正收益;batch1 最佳 1.0739x。⚠️ 注意:**这是旧模型**,新 `layout_aware` 模型下尚不能自动继承。
2. **静态稠密迁移扫描**(条件能力,未标定):最佳 **B2/i128/o256 用 PIM QKV+全 O投影**(1.4295 warm/1.2949 cold);B2/i128/o4096 →1.4012/1.3935 cold;**最大收益来自 QKV/O投影,不是 FFN**。
3. **在线原生 512 过载 trace**(qwen-coder):CHIME 基线与 joint 1.098x、吞吐+9.8%、P95−8.4%。**注意:当时 joint 用 LPT、基线用 CHIME 配对,归因混淆;且 LPT 已退役**。
4. **在线低负载稳定 trace(对齐式组织,唯一正式在线三臂)**:`qwen-coder-stable-512.csv`(时间轴×58.3,≈77 tok/s=0.7×容量),`results/full-joint-qwen-coder-stable-alignment/`:

| 臂 | 吞吐 | E2E均值 | E2E p95 | 平均TPOT |
| --- | ---: | ---: | ---: | ---: |
| chime_baseline | 72.18 | 84.75s | 261.1s | 107.4ms |
| control(对齐式) | 72.18 | 84.65s | 260.8s | 107.3ms |
| joint(对齐式) | 72.53 | 83.69s | 256.8s | 106.7ms |

joint vs control **只 +1.15%均值/+1.51% p95/+0.49%吞吐**,代价 **976 次 KV 切换 + 358 次算子切换**。机制:平均并发 7.6 时 GPU 路径利用率已 0.97,收益公式的"GPU 空闲容量"乘子≈0。**低负载甜区薄收益;大批量(batch≥16)的收益在更高并发/长输出区,单副本设备服务不到的区间**。这是旧模型下最诚实的结论。
5. **M0–M4 静态异质联合矩阵**(`segmented.py` 管线):**joint==attention(收益全是 KV 落点,算子计划贡献为 0)、free 搜索==真实搜索(大批量区最优时机对成本模型不敏感)、batch16/5200/8192 复现 1.503x(闭包 `_kv_closure` 救回可能)、batch8/8192/8192 记录为诚实败例(+18% 输给自己 best_static)**。
6. 在线 joint.json 决策分解:527 决策里约 90% 有算子放 PIM、主力是 O投影(部分比例),QKV 只有 2 个时刻(与 KV 迁移 XOR 互斥),FFN 15 个时刻。**在线确实触及算子迁移,但该负载(长上下文 5–10k、中批量)正好落在算子迁移的无效区(短上下文+极低批量才有效)**——这条与下方 8 相互印证。

### layout_aware(旧 `layout-v1` 输出，仅用于历史诊断)

7. **validation 配对实验**(`results/layout-afd-validation-v1/`,15 对全 completed、token 守恒;B4 闭批、每请求输出 32 token、输入 4K/32K/100K、共享 0%/50%;`layout_fixed` vs `layout_greedy`):
   - **默认参数下大部分 0% 收益、0 动作**;
   - 唯一例外:`100K 输入 / 50K 共享前缀 + attention 带宽系数 0.5` → **+33.71%**(TPOT 308.9→231.0 ms;执行 1 次共享历史 KV 拷贝 + 1 次区间调整;准备暴露 0.98 s;搜索墙钟 2.13 s 但不计入模拟时延)。**带宽系数到 0.6 收益消失 → 结果对成本假设极其敏感,不能表述为真实硬件收益**。
   - 在线错峰 4 请求(两个 100K 共享 50K 前缀 + 短请求):默认与 0.5 系数下 AFD 均 0 动作。
8. **形状主矩阵**(`results/layout-shape-matrix-v3-adaptive-summary.csv`,13 形状,按 `AFD_SHAPE_VALIDATION_PLAN.md` 的 H/Δ 规则:输入≤256→H64;≤4096 或 B<8→H128;其余 H256;Δ=min(512,max(128,64B));候选筛选 `phase-screen-v2`;**截至 2026-10-12 重跑汇总器 = 8/13 完成**,缺 b8-i8192-o8192、b16-i5200-o4096/o8192、b32-i2048-o4096/o8192(这 5 个是长输出收益区,部分 `layout_fixed` 已完成，`layout_greedy` 均未完成):

| 形状 | gain | actions | 说明 |
| --- | ---: | ---: | --- |
| b2-i128-o256 | **+29.13%** | 2×区间 | TPOT 103.7→80.3ms;无写侧/无拷贝 |
| b2-i128-o4096 | **+33.06%** | 2×区间 | TPOT 103.7→78.0ms |
| b4-i5200-o4096 | 0% | 0 | |
| b8-i8192-o1024 | 0% | 0 | |
| b8-i8192-o4096 | 0% | 0 | |
| b16-i5200-o256 | 0% | 0 | |
| b16-i5200-o1024 | 0% | 0 | |
| b32-i2048-o1024 | 0% | 0 | |

   **这是新模型第一个正收益信号:收益出现在 B2 极短输入(i128),由算子区间迁移驱动(PIM 区间调整),与旧模型稠密扫描"收益在短上下文+极低批量、来自 QKV/O投影"跨后端互相印证。** B16/B32 长上下文区目前只在 o256/o1024(0%)上确认,o4096/o8192 未完成——**还不能论断新模型下长上下文区没有收益**。
   - 备注:若读取 summary 时只有 3 行,那是旧快照,用 `python3 -B -m tools.summarize_layout_shape_matrix` 重跑刷新。
9. **资质问题(诚实记录)**:`run_layout_shape_matrix.py` 默认 `--config=configs/layout-aware-long-context.yaml`(文档标注"合成压力场景"),因此主矩阵 manifest 的 `source_config` 指向压力配置;运行语义上工作量被任务原子覆盖(不影响数值),但**记录会误导审稿人,建议把 `source_config` 改为中性基础配置或注明"仅继承硬件成本参数"**。另外 `tools/summarize_layout_shape_matrix.py` 直接运行会因相对导入报错,必须 `python3 -m tools.summarize_layout_shape_matrix`。

## 代码库地图

- **core**:`elasticafd/simulator.py`(迭代级事件模拟 + `run_simulation` 按 `cost_backend` 分派)、`cost.py`(roofline+96 层事件 DAG+子批划分)、`dense_offload.py`、`segmented.py`、`joint.py`/`policy.py`、`state.py`、`config.py`(新增 layout 参数与校验)、`specs.py`、`chime_baseline.py`、`capacity.py`、`pipeline.py`、`partial_softmax.py`(显式 Merge 规范化)。
- **layout_aware**:`elasticafd/layout_aware.py`(PrefixCatalog/KVBlock/PhysicalCopy/WeightCopy/PimInterval/LayoutWorld/StageCost/evaluate_step/LayoutAction/generate_candidates/choose_action/run_layout_simulation)。
- **runners**(tools/):`run_full_joint_trace.py`(四模式 control/attention/f_all/joint)、`run_full_joint_512.py`、`rescale_online_load.py`、`run_heterogeneous_joint_matrix.py`、`prepare_static_recipes.py`、`prepare_burstgpt_trace.py`;**新**:`run_layout_scenarios.py`、`run_layout_sensitivity.py`、`run_layout_afd_validation.py`、`run_layout_shape_matrix.py`、`summarize_layout_shape_matrix.py`。
- **traces**:`workloads/static/`(同质锚点+异质剖面)、`workloads/public/`(BurstGPT 窗口、qwen-coder 系列、mooncake/qwen-thinking 候选)。
- **结果**:`results/full-joint-qwen-coder-stable-alignment/`、`heterogeneous-joint-matrix-recheck{,-extra,-freeb32}/`、`layout-afd-validation-v1/`、`layout-shape-matrix-v3-adaptive*/`(含 preflight 目录,仅性能预检不纳入收益表)。
- **文档**:`CURRENT_STATE.md`(当前状态权威记录,含 2026-10-11 layout_aware 指针)、`LAYOUT_AWARE_MODEL.md`(新模型输入/假设/成对命令)、`AFD_LAYOUT_VALIDATION.md`、`AFD_SHAPE_VALIDATION_PLAN.md`、`docs/` 各实验汇总、各 RUN 报告、`PARAMETER_PROVENANCE.md`。
- **测试**:合计 **202 项通过**(2026-10-12 `python3 -B -m unittest discover -s tests` 实测；其中 34 项 `test_layout_aware`);`results/` 被 gitignore。

## 设计文档 vs 代码差距(2026-10-12 更新)

核心机制**已在新后端落地**:M_p 共享聚合、KV 块表 + 引用计数 + 就绪状态、连续 PIM 区间 + 显式 Merge、动态调整动作族(写侧切换/历史拷贝/区间调整)、短窗口收益评价。剩余差距:
1. **未标定大头**(最大的承力面):`layout_pim_dense_scale=1.0`、`layout_pim_softmax_gelements_s_per_rankset=50`(无实测 PIM softmax 时的临时值,也有别的 proposed value)、`layout_boundary_overlap_fraction=0.25`、`layout_preparation_packet_bytes=4194304`(后台 DMA 暂定抢占粒度)、`link.transfer_latency_us=0`(待测链路启动开销)、`layout_weight_conversion_us_per_gib=0`(乐观下界)、`layout_rankset_sync_us=0`、`layout_survival_mean_tokens=256`。**validation 已在 |0.5↔0.6| 带宽系数上看到收益跳变,标定前任何硬件结论都无效**。
2. **在线/性能**:`policy_search_wall_time_s` 单独报告、不计入模拟时延——慢搜索不代表 online-ready(文档自己写明;100K 场景搜 1.8–4.3 s,B2-o4096 形状搜到 ~118 s)。在线效果须单独核对决策时间预算。
3. **覆盖盲区**:稀疏 `attention_selector=recent_blocks`、异质/共享扩展、其他 B16/B32 形状、12/24 候选预算敏感性、近端瓶颈(共享前缀未分配 DIMM/物理 rank)均**待运行**。B16/I5200/O4096 与 B32/I2048/O4096 已完成。
4. legacy→layout_aware 差异归因未做:为什么 legacy 的 KV 落点收益(1.5x)在新框架短输出区全部消失、只有 B2 区间迁移收益闪现——是成本结构还是策略覆盖,值得专项诊断。

## 先前停机要求与诊断（2026-10-12 历史记录）

用户当时要求停止所有实验，先定位旧有效形状在新框架里保持固定 AF 的原因。旧 `layout-v1` 有 8/13 对完成，另 5 对中断，不能把未完成行当作 0% 收益。后来用户明确要求按旧形状运行新版成对实验；上述停机要求已由新请求解除。最新完成结果见 `AFD_SHAPE_VALIDATION_PLAN.md`。

首要旧版问题：`_stage_cost` 把不同层、甚至同层有依赖的不同算子跨 rankset 取最大值。`layout-v2` 只在同一算子调用内取组间最大值，依赖算子与跨层周期顺序相加。其次，旧版 `layout_survival_mean_tokens - generated` 会把已生成 256 token 的活跃请求预测为近乎结束；现在改为对配置的 Weibull 总长度先验作条件存活预测，默认形状 1 为指数先验。另修正预测时钟、稀疏窗口、准备链路排队、H/Δ 增长预留、多副本 GPU 资源分片及无引用在途前缀释放，并记录每个候选得分或拒绝原因。旧版 B8/I8192/O4096 初始单子批 GPU/PIM 约 51.89/13.41 ms；**新版单轮诊断**约 51.90/53.62 ms，瓶颈转为 PIM。B32 完整实验进一步暴露了多个 GPU 写入侧需同时切换才能满足增长预留的问题，现已补组合容量恢复；34 项针对性与全量 202 项单元测试通过。

## 挂起的下一步(历史记录；当前不要执行)

1. **历史计划：补主矩阵缺的 5 个长输出形状**(b8/i8192/o8192、b16/i5200/o4096/o8192、b32/i2048/o4096/o8192)。当前因层间时序问题暂停，不能直接继续。
2. **验证 B2 正收益的机制**:b2/i128/o256、o4096 的 +29~33% 由区间迁移驱动——确认它对应旧稠密扫描的 QKV/O投影搬移,并做 `layout_pim_dense_scale`/`layout_pim_softmax_scale` 敏感度,判断这个收益是"机制真实现"还是"未标定成本下的假象"。
3. **未标定参数敏感度扫描**:`run_layout_sensitivity.py` 逐项扫 pim_dense/softmax/边界/准备暴露/权重转换。
4. **修复记录诚实性**:`run_layout_shape_matrix.py` 默认 config 改为中性基础配置(或 manifest 注明仅继承硬件参数)。README 里旧版测试描述若与 191 不符需更新。
5. **在线验证**:把 `layout_aware` 后端搬到真实到达 trace(qwen-coder 系列),核对 `policy_search_wall_time_s` 满足在线决策预算。
6. legacy→layout_aware 归因诊断(见差距 4)。
7. 历史待办：当时 `layout_aware` 改动尚未入库、GitHub 只有 legacy；2026-10-14 同步时已纳入本次提交。

## 新对话如果只有一句话,这个方向最该问

> "`layout-v2` 已修正跨层与跨算子依赖、条件存活预测，并补上 GPU 增长预留的组合容量恢复。B8、B16、B32 三个旧形状已完成新版完整配对，其中 B16 和 B32 分别得到模型估计 +33.10% 和 +32.62% 吞吐收益；成本未标定，策略搜索墙钟未计入模拟延迟。"

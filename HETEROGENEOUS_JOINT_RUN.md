# 异质静态联合矩阵 (M0–M4) 运行手册

本流水线已交付(代码侧),尚未在本 CLI 上完整跑过实验。运行者执行下方命令后,
把 `results/heterogeneous-joint-matrix/` 的结果交给模型审查,再由模型更新
CURRENT_STATE —— 现在写入的是**机制预期**,不是已观测结论。

## 1. 交付物与入口

| 文件 | 角色 |
| --- | --- |
| `tools/prepare_static_recipes.py` (M0) | 生成静态配方库(17 锚点 + 3 profile × 2 种子 × 3 batch = 18 异质),容量预检 |
| `elasticafd/segmented.py` (M1-A) | 分段联合规划引擎:`DCSchedule` / `state_views` / 合法性谓词 / `SwitchPricing` / `eval_schedule`(running-peak)/ `search_operator_timeline` / `search_kv_timeline` / `search_joint` |
| `tools/run_heterogeneous_joint_matrix.py` (M1-B+M2+M3 跑者) | 对角跨乘(scenario × formation × strategy × arm),内生独立 best_static 对照 + fresh-engine 重放守卫 |
| `tools/analyze_heterogeneous_joint_matrix.py` (M4) | 消化 `matrix.csv` → `summary.json` + `HETEROGENEOUS_JOINT_SUMMARY.md` |
| `tests/test_segmented_planning.py`, `tests/test_static_recipes.py` | M0/M1-A/B/M3 单元回归(35 项) |

执行入口统一在仓库根:

```bash
# 单元回归(约 3 分钟)
python3 -B -m unittest tests.test_segmented_planning -q
python3 -B -m unittest tests.test_static_recipes -q

# 冒烟(单个配方, 分钟级)
python3 -B -u tools/run_heterogeneous_joint_matrix.py \
  --only anchor_b2_i128_o256 --formations lpt --arms real \
  --output results/heterogeneous-joint-matrix-smoke

# 推荐先导(anchors=quick: 5 个标志形状; 异质全 profile; 默认成为 lpt+alignment;
# 六策略全跑, 双经济臂) —— 这是论文叙事的核心证据
python3 -B -u tools/run_heterogeneous_joint_matrix.py \
  --anchors quick \
  --output results/heterogeneous-joint-matrix

# 全量(17 锚点 + 18 异质 × 2 formation × 2 arm, 数小时)
python3 -B -u tools/run_heterogeneous_joint_matrix.py --anchors all

# 汇总(在跑完后)
python3 -B tools/analyze_heterogeneous_joint_matrix.py \
  --matrix results/heterogeneous-joint-matrix/matrix.csv
```

`--dry-run` 只打印配方/单元计划,不产生结果。`--limit N` / `--only 子串` /
`--only-batch 8` 用于分片续跑;失败的场景不会污染跨版本缓存(每次场景独立
fresh engine,`schedules.json` 记录 runner 代码哈希与全部 violations)。

`--report-arms`(默认等于 `--arms`)把「搜索臂」与「报告臂」解耦:搜索只在
`--arms` 下跑一次,每张 schedule 再在每个 `--report-arms` 下重打分。缺省时等于
原 cartesian 乘积;显式传 `--arms free --report-arms real` 可以在只做 free 搜索、
不重跑昂贵 real 搜索的前提下,补出 M3 经济轴所需的 `(search=free, report=real)`
反事实行。三个 M1-B 守卫(joint_static_reference、best_constant_menu、
fresh-engine 重放)不依赖命令行臂集,始终按 real 价格计(2026-09-24 修复:
此前 `--arms free` 单独跑会 `KeyError: 'real'`)。

## 2. 矩阵定义

```text
scenario   —— 17 个同质锚点 + 18 个异质配方(见 workloads/static/manifest.json)
formation   —— subbatch_scheduler ∈ {lpt, alignment}
strategy    —— control / attn(仅未来-KV 搜索)/ f_only(仅算子时间线)/
               joint(两相搜索)/ ffn_pilot / qkv_pilot
cost arm    —— real(100us/切换 + replica delta 权重拷贝 @ 链路带宽)
               与 free(全零)。每张 schedule 都在两个 arm 下打分:时间线选择
               反映 arm,报告也体现其反事实。
```

搜索默认:`decision_interval=64`、`lookahead=256`、`min_residence=128`、
`max_segments=3`、生命周期采样上限 96 段(stride=horizon/96)。

**可行性闭包(2026-09-24 加入)**:`search_kv_timeline` 的逐 epoch 贪婪只按
`[token, token+lookahead)` 短窗口验收迁移;而 GPU-KV 在请求结束前单调累积,贪婪
会在大 batch × 长输出形状把未来 KV 迁过头、在生命周期尾部打穿 GPU 314 GiB 容量
(clip 到 t≈4450 的旗舰形状),返回 full-lifetime 不可行的 schedule。`_kv_closure`
在搜索收尾时检查全生命周期可行性,不可行则对崩溃块 GPU-tail 的请求二分求最大的
可行「转回 PIM」边界 —— 即把贪婪结果修复成有界前缀窗口(在 B16/I5200/O8192 上
从 infeasible 恢复到 1.50x,对齐 trusted grid 的 1.5074x)。该缺陷由 M1-B 守卫以
violation 形式捕获后定位修复;`tests/test_segmented_planning.py` 的两个闭包回归
测试(37 项单测)固化此行为。

## 3. M3 成本接入语义(重点)

- `switch_execution_us=100`:每次 KV writer 变更 + 每次算子计划变更各计一次,
  与 512-request 配对的 `switch_execution_us` 一致。
- **replica/常驻-reserve-max** 语义:PIM 权重只涨不缩、切换路由不换权。搜索将
  "峰值常驻"作为决策变量参与容量(running-peak);只有 reserve max 增长才在链路
  上以 60 GB/s 拷贝 delta,首个事件含每层 DMA 启动延迟。
- `segment_setup_us` 为信息性字段:512 运行的每层段设置已含在单迭代事件成本内,
  再计一次属重复计费 —— 在你提出疑问的那轮解释中已确认取"前者"(replica),
  不重复收段设置。
- `free` 臂把三者全部置零,是"选择时机本身受成本影响"的敏感对照。

## 4. M1-B 对照与守卫

每次场景 × formation:

1. **主对照(全工况,廉价)**:全 DC 合法菜单上的 best *恒定* 计划(含 qkv/ffn
   组合,是 JointAF 只优化 O_proj 分数子空间的严格超集),断言
   `segmented[joint] ≤ best_constant + ε`。"分段不得输给不分段但聪明的冻结计划"。
2. **独立规划器(次对照)**:`JointAFEventModel`(device_contiguous,
   split_merge_owner=pim)上用 `JointAFScheduler.run_best_static` 冻结出
   (KV 指派 + 一个 O_proj 分数) 的 best-static,映射为 `DCSchedule`,在**同一个**
   事件引擎上以 real 价格重打分。因该指派空间随 batch 指数增长,
   `run_best_static` 现在真正尊重 `max_cost_evaluations` budget(runner 用 128);
   `reference_partial` 记录预算是否触顶(小 batch 仍全枚举,大 batch 为"预算内
   best-static",如实标注,不做未触顶时的过度解读)。
3. 最好的 joint schedule 在**第二个全新引擎**(同 formation 配置)上重放,断言
   total 完全相等(防止 measure 缓存污染跨场景比较)。

任何违反都写进 `schedules.json["violations"]`,汇总页单独列出 —— 若 segmented
在某个单元格输给任一参照,或重放不一致,这是需要如实上报的真问题,不是你该
忽略的噪音。

## 5. 已知的诚实性边界(防止过度解读)

- 搜索是确定性有界搜索,不是无界 oracle;`max_segments=3` 意味着 ≥4 段的复杂
  时间线不会被发现。
- `lpt`/`alignment` 分区内核复用已验证的实现;它们是对齐行为,pairing 的
  "预测"用共享成本预测器完成,不是 CHIME 运行时训练的随机森林 —— 不是官方
  artifact 复现。
- 容量预检含 1 GiB activation workspace 裕量;确定性预检通过不代表每块中点是
  唯一可行点(运行中逐块 running-peak 才是权威)。
- 异质配方是把 long-context-first 排好序的闭批;它测的是"负载突变切换"能力,
  不是在线到达率语义(在线/到达率是另一个实验,未在本流水线内)。

## 6. 交付检查清单(全部完成,本会话)

- [x] M0 生成器 + 35 配方(确定性、manifest、.metadata.json、容量预检全过)
- [x] elasticafd/segmented.py(M1-A 引擎;primitive + 单测 + fresh-engine 真对照)
- [x] run_heterogeneous_joint_matrix.py(M1-B 对照、M2 轴、M3 臂、守卫)
- [x] analyze_heterogeneous_joint_matrix.py(M4 汇总)
- [x] 单测 35 项全过(含修复:dual-def、boundary-0 cut 丢失、零宽 tail、
       running-peak 端点集合、cost((0,plan)) 括号、qkv 合法性与事件模型对齐)
- [x] 旧 kv 切换计数、real/free 双打分、replica delta copy 语义按交接参数落地
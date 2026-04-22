# RL风格放宽FJSP求解器技术报告

## 1. 报告目的

这份报告说明当前求解器是如何从以下三份材料逐步落地出来的：

- [输入输出结构.md](F:/huawei_fjsp_llm/huawei_fjsp_llm/输入输出结构.md)
- [问题描述文档.md](F:/huawei_fjsp_llm/huawei_fjsp_llm/问题描述文档.md)
- [rl1.md](F:/huawei_fjsp_llm/huawei_fjsp_llm/rl1.md)

报告分成三部分：

1. 技术报告：业务约束如何转成代码、解是如何生成的、结果是怎样得到的
2. 参数说明：每个参数控制什么偏好、为什么存在、是怎么调出来的
3. 代码说明：当前 `scripts/rl_relaxed_solver.py` 的结构、关键函数和数据流

这里先明确一个关键事实：

- 当前实现是“RL 风格求解器”，不是“训练了 PPO/DQN 神经网络的完整强化学习系统”
- 它遵循了 `rl1.md` 里的状态、动作、奖励设计思路
- 但为了先拿到稳定、可解释、可复验、可快速迭代的合法解，最终落地成了“奖励塑形驱动的启发式 dispatch + 两阶段修复”的工程版实现

这不是偷换概念，而是一个有意识的技术折中：先把奖励思想做成稳定可控的调度策略，再逐步向真正的学习型策略扩展。

## 2. 原始文档是怎样转成求解问题的

### 2.1 `输入输出结构.md` 提供了什么

这份文档主要提供了两个东西：

- 输入 JSON 的完整结构
- 输出 JSON 的合法格式

输入里最关键的字段是：

- `time.current_time` 和 `time.current_date_time`
- `config.max_output_horizon`
- `eqp` 里的设备厂区和停机区间
- `task` 里的任务、路径、工序、候选机器、qtime、批属性
- `transition` 里的跨机转运时间
- `setup` 里的顺序相关切换时间

输出里最关键的约束是：

- 每个任务要给出选中的 `path_id`
- 每个工序要给出 `temp_machine_id`
- 每个工序要给出明确的开始结束时间

因此，代码的最基本职责就变成了：

1. 从输入 JSON 中抽出可调度对象
2. 为每个任务选路径、为每道工序选机器、定开始时间
3. 输出成文档要求的 JSON 结构

### 2.2 `问题描述文档.md` 提供了什么

这份文档定义了真正的业务约束，也就是“什么叫合法解”。当前实现中直接落地的约束包括：

- FJSP 机器可选性：工序只能放到自己的候选机器上
- 释放时间：任务不能早于 `earliest_ava_time`
- 工序前后顺序：后工序不能早于前工序完成加转运之后
- 最大最小时间间隔：由 `qtime_info` 控制
- 跨厂转运限制：只有允许的 `diff_factory_info` 才能跨厂
- 设备维修窗口：不能与 `eqp_down_interval` 重叠
- 顺序相关 setup：同机相邻两道非组批工序之间要留 setup 时间

此外，文档还强调了：

- 路径可以多选一
- setup 是非对称的
- batch 工序和普通工序是不同处理模式

### 2.3 `rl1.md` 提供了什么

`rl1.md` 没有直接给现成代码，但它给了非常重要的建模方向：

- 状态应该包含时间、设备状态、待调度工序、已完成产量、setup 次数
- 动作应该是“选工序、选机器、决定何时开始”
- 奖励应该同时考虑产量、setup、按时率、约束惩罚、组批奖励

这直接影响了当前代码的两个核心设计：

1. 候选动作不再只看“谁最早能开工”，而是看“谁的综合奖励最高”
2. 参数不写死在逻辑里，而是显式暴露成 `score-*` 和 `phase2-*`，方便按目标调参

## 3. 当前版本的合法解定义

你后来给了一个非常关键的放宽定义：

- 满足所有约束
- 但“组批机器视为无限产能”

因此当前求解器求的是“放宽组批容量后的合法解”。这一定义下：

- 非组批工序按真实单机串行资源处理
- 组批工序不参加真实容量竞争，只要它本身的前后约束、路径约束、时间约束满足，就允许被直接推进

这带来的好处是：

- 能快速覆盖完整工艺链
- 不会因为 batch 组合决策过于复杂而卡住全局排程
- 能先把主要难点放在非组批瓶颈机台、setup、qtime 和跨厂限制上

代价也很清楚：

- 这不是原题的完整 batch 资源模型
- 因此这里的“合法”是你指定的放宽版合法，而不是原始业务模型下的完全精确合法

## 4. 从文档到代码的实现路线

## 4.1 大体架构

当前主程序是 [rl_relaxed_solver.py](F:/huawei_fjsp_llm/huawei_fjsp_llm/scripts/rl_relaxed_solver.py)，整体分成六层：

1. 工具函数层：时间解析、数字转换、路径签名
2. 数据建模层：`TaskSpec`、`ProcessSpec`、`MachineSpec`、`InstanceData`
3. 实例构建层：从大 JSON 流式解析出任务、设备、转运、setup
4. 调度决策层：`RelaxedRLScheduler`
5. 结果输出层：写出标准 JSON
6. 合法性校验层：对生成解做全量验证

## 4.2 为什么用流式解析和缓存

原始数据规模不小：

- 任务数接近两千
- 工序数接近一万
- setup 矩阵是稀疏但很大的一张表

如果每次都整文件反复读、setup 整体进内存，会很慢，也不稳定。因此代码做了两个工程化处理：

- 用 `ijson` 流式解析大 JSON
- 把 setup 行存成 SQLite 压缩行存储

这样做的原因很直接：

- 任务和设备结构可以一次解析后缓存成 pickle
- setup 可以按需逐行读取，不必整表常驻内存

这部分后来还加了“输入签名校验”，避免换算例时误复用旧缓存。

## 4.3 为什么先固定一条路径

理论上每个任务可能有多条 `process_path`，真正的最优做法是路径选择和调度一起做联合优化。但如果一开始就这么做，动作空间会明显变大。

因此当前版本先采用了一个务实策略：

- 先为每个任务选一条“乐观成本较低”的路径
- 再在这条路径上做机器选择和排程

路径选择准则是：

- 非组批最短加工时间更重要
- 组批最短加工时间次之
- 累计最小等待时间再作为补充

原因是当前目标里真正决定产量和 setup 的，主要还是非组批机台上的主节拍。

## 4.4 状态是怎么设计的

`rl1.md` 里建议状态包含时间、设备、待调度工序、已完成情况。当前代码把它具体落成了这些结构：

- `machine_free`：每台非组批设备何时可用
- `machine_last_proc`：这台机上最后一道工序是谁
- `machine_last_family`：这台机上最后一个“工序家族”是什么
- `next_idx`：每个任务当前排到第几道工序
- `task_records`：每个任务已经排了哪些工序
- `proc_records`：每个工序的实际排程结果
- `task_status`：任务是 `active/deferred/done/infeasible`
- `setup_count`：累计正 setup 次数

这就是一个典型的调度 MDP 的工程化状态表示。

## 4.5 动作是怎么定义的

完整 RL 里动作会写成：

- 选哪个任务的下一道工序
- 放到哪台机器
- 什么时候开始

当前代码里，这个动作被简化成：

- 对每个“当前可排的任务下一道工序”，枚举其候选机器
- 对每个“任务-机器”组合计算一个 `CandidateEval`
- 从这些候选动作里选分数最高的一个落地

`CandidateEval` 里包含：

- `start/finish`
- `setup_time`
- `est_final`
- `upper_bound`
- `option_priority`
- `started`
- `same_family`

也就是说，动作不是一个裸选择，而是一个“带评估信息的候选排程动作”。

## 4.6 约束是怎么提前嵌进动作可行性里的

当前代码不是“先乱排，再统一修约束”，而是把大部分硬约束都放进候选动作生成阶段。

核心函数是：

- `compute_bounds`
- `transfer_allowed`
- `fit_after_maintenance`
- `has_forward_compatibility`

其中：

- `compute_bounds` 负责释放时间、前序完成、转运、qtime 最小最大间隔
- `transfer_allowed` 负责跨厂合法性
- `fit_after_maintenance` 负责避开维修窗口
- `has_forward_compatibility` 做一步前瞻，避免当前工序选了机器后，下一道工序根本没有合法去向

这一步前瞻虽然很轻量，但很关键。它显著减少了“当前看可行、后面死路”的假可行动作。

## 5. RL 思想是如何落成当前调度策略的

## 5.1 为什么没有直接上 PPO/DQN

从研究角度，PPO、DQN、GNN+RL 都可以做。但当前项目一开始的目标不是写论文，而是先拿到：

- 稳定
- 合法
- 可解释
- 可以快速反复调参
- 很快产出完整解

真实强化学习训练要先解决：

- 状态编码
- 合法动作掩码
- 稀疏奖励
- 大实例采样效率
- 训练稳定性
- 推理时仍然要保证强约束不被破坏

这条路不是不能走，而是首版落地成本高、调试周期长，而且很容易先卡在“训练得动”而不是“解能交付”。

所以当前版本采用了一个折中方案：

- 用 RL 的奖励思想来设计评分函数
- 用启发式 dispatch 来做每一步动作选择
- 用两阶段流程兼顾 horizon 产量与完整性

本质上，这是“reward-shaped greedy policy”。

## 5.2 第一阶段：面向 horizon 的产量最大化

第一阶段只关心一个核心目标：

- 在 `max_output_horizon` 之内尽量多完成高产量任务

具体做法是：

1. 先把所有能直接推进的 batch 前缀工序推进
2. 对所有可行的非组批候选动作打分
3. 在最早开始时间附近做一个 `lookahead` 窗口
4. 从窗口内选综合得分最高的候选

第一阶段对“新开任务”是保守的：

- 如果一个任务从头开始都几乎不可能在 horizon 内完工，就先 `deferred`

这样做的原因是避免早早占掉关键机台，却拿不到 horizon 内产量。

## 5.3 第二阶段：补全完整解

如果只做第一阶段，往往会得到一个“horizon 内产量不错，但有些任务没排完”的解。但你明确要求：

- 每个解都必须包含所有工件
- 必须是完整解

所以求解器后面又加了第二阶段：

- 把第一阶段被 `deferred` 的任务重新激活
- 把重点从“horizon 产量最大化”切到“补齐所有剩余工序”
- 评分里更强调延续已开始任务、减少额外 setup、优先走高密度任务

第二阶段不是为了提高 horizon 指标，而是为了满足完整性和合法性。

## 5.4 最后一步：修复不完整任务

在复杂约束下，即便做完两阶段，也可能剩下少量任务因为局部状态原因没有被补完。于是代码最后再做一次：

- `repair_incomplete_tasks`

它会：

- 先保留已经完整的任务
- 重建机器状态
- 再把不完整任务补排

这一步的目的不是优化指标，而是确保“最终输出一定是完整的、可校验的解”。

## 5.5 奖励是如何落成评分函数的

`rl1.md` 建议奖励同时考虑产量、setup、约束、组批、时间惩罚。当前代码把这个思想压缩成了两套评分函数：

- `score_candidate`
- `score_candidate_phase2`

第一阶段评分主要包含：

- `task.weight * score_weight`
- `density * score_density`
- 已经开工过的任务奖励 `score_started`
- 同家族连续加工奖励 `score_family`
- qtime 紧迫奖励 `q_bonus`
- 正 setup 固定惩罚 `score_setup_fixed`
- setup 时长线性惩罚 `score_setup_per`
- 晚开工惩罚
- 候选机器优先级惩罚
- 预计完工时间惩罚

第二阶段评分逻辑类似，但权重不同，更偏向：

- 延续已经开始的任务
- 少加 setup
- 快速补齐全量任务

也就是说，当前参数本质上就是“奖励函数权重”。

## 6. 约束落地明细

下面把“文档里的约束”和“代码里的位置”对起来。

### 6.1 释放时间

来源：

- `task.earliest_ava_time`

实现：

- `compute_bounds` 的初始下界

效果：

- 第一工序不能早于任务释放时刻

### 6.2 工序先后顺序与转运

来源：

- 前序工序结束时间
- `transition[from_eqp][to_eqp]`

实现：

- `compute_bounds`

效果：

- 后一道工序最早开工时间至少等于“前工序完工 + 转运时间”

### 6.3 跨厂转运限制

来源：

- `diff_factory_info`

实现：

- `transfer_allowed`

效果：

- 前后两工序如果跨厂，必须在允许列表中

### 6.4 最小/最大时间间隔

来源：

- `qtime_info`

实现：

- `compute_bounds`

效果：

- 把 qtime 最小约束转成开始时间下界
- 把 qtime 最大约束转成开始时间上界

### 6.5 设备维修时间

来源：

- `eqp_down_interval`

实现：

- `fit_after_maintenance`
- `validate_solution`

效果：

- 候选动作会自动跳过维修区间
- 最终验证时再次检查是否重叠

### 6.6 机器可选性

来源：

- `process.eqp_list`

实现：

- 候选动作只在 `proc.candidates` 中枚举

效果：

- 工序只能落到文档允许的机器上

### 6.7 顺序相关 setup

来源：

- `setup[from_process][to_process]`

实现：

- 非组批工序使用 `machine_last_proc` 查 setup
- `record_schedule` 统计正 setup 次数
- `validate_solution` 再校验相邻非组批工序是否留足 setup

效果：

- setup 不但影响开始时间，还进入优化目标

### 6.8 完整解要求

来源：

- 你的追加要求

实现：

- 两阶段补齐
- `repair_incomplete_tasks`
- `validate_solution` 检查任务是否完整覆盖选定路径

效果：

- 输出必须包含所有任务的整条路径

### 6.9 组批无限产能放宽

来源：

- 你的追加定义

实现：

- batch 工序不更新 `machine_free`
- batch 工序不参与 setup 统计
- batch 工序通过 `advance_batch_prefix` 自动推进

效果：

- 把 batch 机器从稀缺资源调度里拿掉，只保留路径和时间逻辑

## 7. 结果是如何一步步做出来的

## 7.1 初始目标

最早阶段的目标是：

- 在 `24480` 截止前产量达到 `17500`
- 正 setup 次数小于 `1000`
- 并且必须是完整解

当前保留的基线结果是：

- [rl_relaxed_full_solution.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/rl_relaxed_full_solution.json)
- `completed_weight_within_horizon = 18524.28`
- `setup_count_positive = 767`

也就是已经明显超过了 `17500/1000`。

## 7.2 后续优化思路

后续优化主要围绕三件事展开：

1. 提高单位瓶颈时间能带来的产量
2. 尽量让同族工序连续加工，压 setup
3. 控制新任务开启节奏，避免 horizon 前完成不了的任务过早占位

对应到参数上，重点搜索的是：

- `lookahead`
- `score_weight`
- `score_density`
- `score_family`
- `score_setup_fixed`
- `score_setup_per`
- `phase2_*`

## 7.3 当前保留结果

仓库里保留了三组有代表性的结果：

1. 基线完整解
   - [rl_relaxed_full_solution.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/rl_relaxed_full_solution.json)
   - `18524.28 / 767`
2. 更高产量的 `<800` setup 候选
   - [rl_relaxed_candidate_18547_784.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/rl_relaxed_candidate_18547_784.json)
   - `18547.29 / 784`
3. 当前最强的 `<800` setup 候选
   - [rl_relaxed_candidate_18598_791.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/rl_relaxed_candidate_18598_791.json)
   - `18598.93 / 791`

本地还找到了一个更高产量但略超 `800` setup 的版本：

- `18723.73 / 806`

它说明当前策略前沿大致已经到：

- 产量 1.87 万左右
- setup 800 左右

## 7.4 新算例验证

对新算例：

- [input_data.json](C:/Users/ASUS/Downloads/huawei_fjsp_副本/data/data1/input_data.json)

当产量截止时间改为 `23040` 时，当前求解器跑出的完整合法解指标是：

- 全部任务：`1845 / 1845`
- `completed_tasks_within_horizon = 1474`
- `completed_weight_within_horizon = 17855.79`
- `setup_count_positive = 755`
- `error_count = 0`

对应输出文件：

- [input_data_23040_solution.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/input_data_23040_solution.json)

这说明当前框架已经不只适用于单一原算例，也能迁移到新的输入和新的 horizon。

## 8. 参数意义说明

这一节回答两个问题：

- 这个参数控制什么
- 这个参数为什么要有

还要补一个非常重要的前提：

- 这些参数不是通过梯度训练自动学出来的
- 它们来自“RL 奖励思想 + 大量实验搜索”的人工调参

也就是说，它们本质上是“奖励塑形权重”和“搜索窗口控制量”。

## 8.1 运行与工程参数

### `--input`

作用：

- 指定输入算例 JSON

为什么要有：

- 求解器不能只绑死在 `data/data1` 目录
- 不同算例需要显式切换

怎么来的：

- 来自工程需求，不是优化逻辑本身

### `--horizon-override`

作用：

- 覆盖输入文件里的 `config.max_output_horizon`

为什么要有：

- 你明确提出过“换算例，同时产量统计时间改成 23040”
- 很多时候我们只想改评价窗口，不想改原始 JSON

怎么来的：

- 来自多算例、多 horizon 复验需求

### `--output`

作用：

- 指定输出解文件位置

为什么要有：

- 不同实验要保留不同结果

### `--instance-cache`

作用：

- 缓存实例结构，包括设备、任务、选中的路径等

为什么要有：

- 反复调参时，没必要每次都重新解析整个 JSON

### `--setup-db`

作用：

- 指定 setup SQLite 行存储数据库

为什么要有：

- setup 稀疏矩阵很大，需要独立缓存
- 不同算例最好用不同的 setup 库

### `--rebuild-instance`

作用：

- 强制重建实例缓存

为什么要有：

- 输入文件改了、路径逻辑改了、怀疑缓存脏了时，需要强刷

### `--rebuild-setup`

作用：

- 强制重建 setup 数据库

为什么要有：

- setup 表变了或换了算例时，需要重建

## 8.2 第一阶段评分参数

### `--lookahead`

作用：

- 只在“最早可开工时间附近”的候选动作里做选择

为什么要有：

- 如果在所有远期候选上全局比较，分数会偏向一些看起来收益高、但实际上把当前机器空闲很久的动作
- lookahead 相当于给 dispatch 加一个局部时间窗，防止过度远视

怎么来的：

- 来自调度经验和 `rl1.md` 的时间惩罚思想
- 实验上它是最敏感的结构参数之一

### `--score-weight`

作用：

- 奖励高重量任务

为什么要有：

- 目标函数直接是 horizon 内完成产量，而不是完成件数

怎么来的：

- 来自 `rl1.md` 的产出奖励

### `--score-density`

作用：

- 奖励单位剩余非组批时间能带来更高重量的任务

为什么要有：

- 单看 `weight` 会偏向大任务
- 单看完工速度会偏向小任务
- `weight / remaining_nonbatch_time` 更接近“瓶颈时间产出密度”

怎么来的：

- 是把“产量最大化”从总量视角改成了效率视角

### `--score-started`

作用：

- 奖励已经开过工的任务继续往下排

为什么要有：

- 减少任务碎片化
- 降低第一阶段把很多任务都开了头但没做完的风险

怎么来的：

- 来自完整解要求和经验观察

### `--score-family`

作用：

- 奖励当前工序和该机器上一道工序属于同家族

为什么要有：

- 同家族连续加工通常更容易得到零 setup 或较小 setup

怎么来的：

- 来自对 setup 表的现象观察
- 是压 setup 的关键参数之一

### `--score-setup-fixed`

作用：

- 只要发生一次正 setup，就给固定惩罚

为什么要有：

- setup 次数本身就是目标之一
- 仅用线性 setup 时长惩罚，不足以压制“很多次小 setup”

怎么来的：

- 对应 `setup_count_positive` 这个离散目标

### `--score-setup-per`

作用：

- 对 setup 时长再做线性惩罚

为什么要有：

- 两次 setup 次数一样时，长 setup 还是更差

怎么来的：

- 补足固定惩罚无法区分 setup 时长的问题

## 8.3 第二阶段评分参数

### `--phase2-started`

作用：

- 第二阶段更强地偏好继续推进已开始任务

为什么要有：

- 第二阶段的主目标是补全完整解，不是再去大规模开新任务

### `--phase2-density`

作用：

- 第二阶段仍然保留对高产出密度任务的偏好

为什么要有：

- 虽然目标转向补全，但仍然希望尽可能不损失 horizon 内产量

### `--phase2-family`

作用：

- 第二阶段继续鼓励家族聚类

为什么要有：

- 修复完整解时若完全不顾 setup，很容易把 setup 指标拉坏

### `--phase2-setup-fixed`

作用：

- 第二阶段对 setup 次数的固定惩罚

为什么要有：

- 第二阶段很容易因为“只顾补任务”导致 setup 暴增

### `--phase2-setup-per`

作用：

- 第二阶段对 setup 时长的线性惩罚

为什么要有：

- 保证补齐时也尽量少做长 setup

## 8.4 为什么要把第一阶段和第二阶段参数分开

这是一个很重要的设计点。

如果只用一套参数，很容易出现两种坏情况：

1. 参数偏向 horizon 产量时，很多任务排不完整
2. 参数偏向完整性时，horizon 内产量掉很多

因此才把策略拆成：

- 第一阶段专注“horizon 内多产”
- 第二阶段专注“把整张解补完整”

对应地，参数也必须拆成两套。

## 8.5 当前默认参数对应什么结果

当前默认参数对应的是仓库里这组结果：

- [rl_relaxed_candidate_18598_791.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/rl_relaxed_candidate_18598_791.json)

默认值是：

```text
lookahead = 80
score_weight = 329.0
score_density = 22850.0
score_started = 140.0
score_family = 365.0
score_setup_fixed = 420.0
score_setup_per = 4.45
phase2_started = 2200.0
phase2_density = 6660.0
phase2_family = 515.0
phase2_setup_fixed = 540.0
phase2_setup_per = 4.45
```

这些值不是“理论最优值”，而是当前实验前沿上最好的折中点之一。

## 9. 代码说明

下面按模块解释当前代码。

## 9.1 工具函数与基础类型

代码开头定义了：

- `to_int`
- `to_float`
- `parse_dt`
- `fmt_dt`
- `normalize_input_path`
- `input_signature`

它们的作用分别是：

- 统一字段类型
- 统一时间字符串格式
- 在分钟时间轴和日期时间字符串之间互相转换
- 为缓存建立“输入文件签名”

随后定义了一系列数据类：

- `CandidateSpec`
- `QTimeSpec`
- `ProcessSpec`
- `TaskSpec`
- `MachineSpec`
- `ScheduledOp`
- `CandidateEval`
- `InstanceData`

这些类的作用是把输入 JSON 转成强类型的内部对象，便于后续调度。

## 9.2 `SetupRowStore`

这个类负责管理 setup 稀疏矩阵。

为什么单独做成一个类：

- setup 非常大
- 但单次调度只会频繁访问少量相邻工序对

它做了三件事：

1. 首次构建时，把每一行 setup 压缩后写入 SQLite
2. 查询时按 `from_proc` 懒加载这一行
3. 用 LRU 行缓存减少重复解压

后来还补了：

- `meta` 元信息
- 输入签名校验

目的就是防止换算例后误用旧 setup 缓存。

## 9.3 `build_task_spec`

这个函数负责把一个任务展开成内部 `TaskSpec`。

关键工作包括：

- 选择路径 `choose_path`
- 生成有序工序序列
- 生成 `seq_to_idx`
- 构建每道工序的候选机器列表
- 抽取 `incoming_qtimes`
- 预计算 `optimistic_total_from`
- 预计算 `optimistic_nonbatch_from`

这两个 optimistic 数组很重要：

- `optimistic_total_from` 用于估计从某道工序往后到任务结束的乐观完工时间
- `optimistic_nonbatch_from` 用于构造产量密度

它们本质上是启发式估值函数的一部分。

## 9.4 `build_instance`

这个函数负责把整个输入 JSON 转成 `InstanceData`。

它会依次读出：

- `current_time`
- `current_date_time`
- `max_output_horizon`
- 全部机器
- 全部转运时间
- 全部任务

然后把结构缓存成 pickle。

这里后来加的关键工程增强是：

- 缓存不仅比较文件时间，还比较输入路径、文件大小、纳秒级修改时间

这是因为仅靠时间戳不足以保证缓存属于当前输入。

## 9.5 `RelaxedRLScheduler`

这是求解器核心类。

它管理整个调度状态，并提供：

- 候选动作评估
- 两阶段求解
- 完整性修复
- 指标统计

### `fit_after_maintenance`

用途：

- 在给定最早开始时间后，找到一个不会撞上停机窗口的可开工时间

### `transition_time`

用途：

- 查前后机器之间的转运时间

### `transfer_allowed`

用途：

- 检查跨厂是否允许

### `compute_bounds`

用途：

- 计算某道工序在某台机器上的时间下界和上界

这是最重要的可行性函数之一。

### `has_forward_compatibility`

用途：

- 做一步前瞻，保证当前机器选择不会让下一道工序完全没路可走

### `process_family`

用途：

- 把工序抽成一个“家族签名”，供 `same_family` 判断

### `evaluate_candidate`

用途：

- 生成并评估一个候选动作

它会综合得到：

- 可行开始结束时间
- setup 时间
- 是否同家族
- 乐观完工时间

### `batch_choice`

用途：

- 给 batch 工序选一个相对合理的机器

因为 batch 被放宽为无限容量，所以这里只需要挑一个满足后续可行性、估计完工较好的选择。

### `record_schedule`

用途：

- 把一个候选动作真正写入状态

这里会更新：

- 任务记录
- 工序记录
- 下一工序索引
- 机器释放时间
- 最后工序
- setup 次数

### `advance_batch_prefix`

用途：

- 把一个任务开头连续的 batch 工序自动推进

这是“放宽组批机器”设计的直接代码体现。

### `score_candidate`

用途：

- 第一阶段评分函数

### `score_candidate_phase2`

用途：

- 第二阶段评分函数

### `solve`

用途：

- 执行完整求解流程

大致流程是：

1. 推进所有任务前缀 batch 工序
2. 第一阶段优化 horizon 内产量
3. 重新激活 deferred 任务
4. 第二阶段补齐完整解
5. 调用 `repair_incomplete_tasks` 做最终兜底

### `metrics`

用途：

- 统计当前解的核心指标

包括：

- 完整排程任务数
- horizon 内完成任务数
- horizon 内完成产量
- 晚于 horizon 完成的任务数
- 正 setup 次数

## 9.6 输出与校验

### `dump_solution`

作用：

- 按文档要求写出 JSON

### `load_solution_records`

作用：

- 从 JSON 解文件反读回内部结构

### `validate_solution`

作用：

- 做最终合法性验证

它会检查：

- 任务是否缺失
- 是否完整覆盖选定路径
- 路径 ID 是否一致
- 机器是否合法
- 加工时长是否匹配
- 是否撞维修
- 是否违反释放时间、前后顺序、转运、qtime、跨厂限制
- 非组批机台上是否发生重叠或 setup 不足

这一步很重要，因为它把“调度器自认为可行”和“最终解文件真实合法”分开做了二次确认。

## 9.7 命令行入口

`parse_args` 和 `main` 负责：

- 接收命令行参数
- 构建实例
- 应用 `horizon` 覆盖
- 构建 setup 存储
- 调用求解或校验

一个典型命令是：

```powershell
python scripts/rl_relaxed_solver.py solve `
  --input "C:\Users\ASUS\Downloads\huawei_fjsp_副本\data\data1\input_data.json" `
  --horizon-override 23040 `
  --rebuild-instance `
  --rebuild-setup `
  --instance-cache cache\input_data_23040.pkl `
  --setup-db cache\input_data_23040.sqlite `
  --output outputs\input_data_23040_solution.json
```

## 10. 当前方法的优点与局限

## 10.1 优点

- 能稳定给出完整解
- 对复杂硬约束支持比较完整
- 可解释，方便调参
- 换算例和换 horizon 比较容易
- setup 和产量之间能做明确折中

## 10.2 局限

- 不是端到端训练出来的真实 RL 策略
- 路径选择仍然是先验固定，不是联动优化
- batch 机器被放宽成无限容量，不是原题全精确模型
- 当前主要是局部 dispatch 决策，没有更强的全局回溯和重排
- 参数目前依赖实验搜索，不是自动学习

## 10.3 如果继续往下做

后续可以沿三条线升级：

1. 真正学习化
   - 用 PPO 或 GNN+PPO 学策略
   - 当前评分函数可以作为初始奖励模板
2. 更强搜索化
   - 加 beam search、LNS、局部交换、重排
3. 更精确建模
   - 恢复 batch 机台真实容量
   - 把路径选择纳入联动优化

## 11. 总结

从你的文档出发，当前代码实际上完成了这样一件事：

- 用 `输入输出结构.md` 定义了读入和输出格式
- 用 `问题描述文档.md` 定义了约束检查和可行调度逻辑
- 用 `rl1.md` 定义了奖励思想和动作评估方向
- 最终实现成了一个“RL 风格、工程可交付、可复验、可调参”的放宽组批 FJSP 求解器

它不是一套学术版完整 RL 训练系统，但它确实把强化学习的核心思想落到了最关键的地方：

- 状态抽象
- 动作评估
- 奖励塑形
- 多目标折中

并且已经产出了稳定可校验的完整解，这就是它当前最重要的价值。

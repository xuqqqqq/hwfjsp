# DeepSeek 策略自迭代 30 轮实验总结

本文档汇总 DeepSeek 驱动的策略自迭代框架在“测试算例1、有限组批、horizon=24480”口径下的 30 轮实验过程、关键结果、有效经验和失败方向。本文只依据当前项目目录中的可复核产物整理，不包含接口密钥或不可复现的外部信息。

## 1. 实验目标与基本口径

### 1.1 实验目标

本轮实验的目标不是单纯追求最高产量，而是在有限组批约束下寻找更低 setup 的可交付完整解。后半段目标明确调整为：

- `completed_weight_within_horizon` 尽量保持在 `185xx` 附近；
- `setup_count_positive` 尽量靠近 `500`；
- 所有解必须完整包含 1751 个任务；
- 所有解必须通过人工编写的有限组批校验器；
- DeepSeek 不允许读取旧解文件、旧输出目录或历史手工调参日志，只能根据本框架传入的文档、prompt、上一轮摘要和校验反馈自我迭代。

### 1.2 算例与校验口径

| 项目 | 内容 |
| --- | --- |
| 算例 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\data\data1\测试算例1.json` |
| 时间窗口 | `horizon=24480`，由算例读取 |
| 组批口径 | 有限组批容量 |
| 完整性要求 | `fully_scheduled_tasks == total_tasks == 1751` |
| 工序数 | `scheduled_ops == 9642` |
| 组批工序数 | `batch_ops == 538` |
| 组批校验器 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\scripts\validate_batch_solution.py` |
| 固定求解内核 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\scripts\strategy_kernel_solver.py` |
| LLM 迭代框架 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\scripts\deepseek_strategy_framework.py` |

### 1.3 合法解判定

本实验中“合法完整解”的判定口径为：

- `error_count = 0`；
- 所有 1751 个任务均被完整排入输出解；
- 工序前后序、释放时间、维修窗口、设备可加工性、qtime 最大/最小间隔、setup 时间、跨厂转运时间均由固定内核生成和校验；
- 组批机器采用有限容量校验：同一批必须满足 family 兼容、批容量约束、批时长一致性、批组机器占用一致性；
- setup 次数只统计 `setuptime > 0` 的正切换次数。

## 2. 框架设计摘要

### 2.1 固定内核与可变策略分离

本实验使用“固定约束内核 + LLM 策略模块”的结构。DeepSeek 不生成完整求解器，也不直接写解文件，只能生成或修改 `deepseek_strategy.py` 中的策略函数。

固定内核负责：

- 读取输入 JSON，构造任务、工序、候选设备、路径、维修、转运、setup、组批约束；
- 过滤所有不满足硬约束的动作；
- 执行两阶段调度主循环；
- 写出完整解文件；
- 调用有限组批校验器输出指标。

DeepSeek 只负责：

- 修改启发式参数；
- 修改路径选择评分；
- 修改第一阶段候选动作排序；
- 修改第二阶段补全动作排序；
- 对每个候选给出规则变更审计与下一轮建议。

这种设计的核心原因是：工业调度问题的约束非常复杂，如果让 LLM 直接生成完整求解器，极容易在 qtime、维修、转运、组批容量等硬约束上出错。因此本实验把“正确性”交给人工编写的内核和校验器，把“规则搜索与参数探索”交给 LLM。

### 2.2 DeepSeek 可修改接口

DeepSeek 每次只能生成一个策略模块，必须包含以下函数：

```python
def describe_strategy() -> str: ...
def describe_rule_changes() -> dict: ...
def get_config(base_config: dict, case_context: dict) -> dict: ...
def score_path(features: dict) -> float | None: ...
def score_phase1(features: dict) -> float | None: ...
def score_phase2(features: dict) -> float | None: ...
```

其中：

- `get_config()` 用于调整超参数；
- `score_path()` 用于在输入文件已有的可选路径之间排序，返回 `None` 表示使用固定内核默认路径规则；
- `score_phase1()` 用于第一阶段 horizon 内产量优先的候选动作排序；
- `score_phase2()` 用于第二阶段补全完整解和降低 setup 的候选动作排序；
- `describe_rule_changes()` 用于记录本候选相对亲本/基线新增、删除、保留了哪些规则。

### 2.3 候选多样性约束

为了避免 LLM 每轮生成同质化策略，框架在 prompt 中强制分配候选探索角色：

| 角色 | 要求 |
| --- | --- |
| `new_dispatch_rule` | 必须在 `score_phase1` 中提出新的派工规则，并删除或禁用至少一个历史无效派工因子 |
| `new_completion_rule` | 必须在 `score_phase2` 中提出新的补全规则，并删除或禁用至少一个历史无效补全因子 |
| `new_path_rule` | 必须在 `score_path` 中提出路径选择规则，或明确删除自定义路径规则并说明原因 |
| `rule_prune_ablation` | 以删规则、消融为主，至少让一个 `score_*` 回退到 `None` 或删除一个奖励/惩罚项 |
| `crossover_prune` | 从精英候选中吸收有效规则片段，同时删除坏片段并做小幅变异 |
| `guarded_parameter_mutation` | 主要调整 `get_config`，但必须说明保留规则、参数变化和下一轮规则建议 |

prompt 中还明确要求：候选之间不能只改文案，至少一个函数的实质逻辑或阈值组合必须不同；所谓“吸收片段”只能吸收策略经验、评分因子、函数逻辑或参数组合，禁止读取、复制或拼接旧解文件中的排程结构。

## 3. 运行产物与数据来源

### 3.1 主要输出目录

本轮实验的全部产物保存在：

```text
F:\huawei_fjsp_llm\huawei_fjsp_llm\outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2
```

关键文件和目录如下：

| 路径 | 作用 |
| --- | --- |
| `final_report.json` | 本次框架最终报告，记录综合评分最优策略和解 |
| `best_summary.json` | 当前 best 的完整指标、参数和规则说明 |
| `population.json` | 框架保留的精英种群 |
| `experience_log.md` | 逐次迭代经验日志，包含每个候选的指标和规则审计 |
| `reports\*.json` | 每个候选的结构化摘要 |
| `solutions\solution_iter*.json` | 每个求解候选输出的排程解 |
| `candidates\deepseek_strategy_round_*_cand_*.py` | DeepSeek 生成的候选策略代码 |
| `prompts\round_*_cand_*_prompt.md` | 每个候选实际收到的 prompt |
| `responses\*.txt` | LLM 原始响应 |

### 3.2 运行命令形态

本目录由多次续跑形成，等价的核心运行方式如下。实际使用时不应在命令或文档中写入 API key，而应通过环境变量提供。

```powershell
$env:DEEPSEEK_API_KEY = "<your-key>"

python scripts\deepseek_strategy_framework.py `
  --input data\data1\测试算例1.json `
  --track finite `
  --output-dir outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2 `
  --max-iters 30 `
  --candidates-per-iter 6 `
  --elite-size 6 `
  --target-weight 18500 `
  --target-setup 500 `
  --setup-penalty 0.2 `
  --solver-timeout 420
```

### 3.3 评分函数口径

由于目标从单纯高产量转为“185xx/500 左右”，框架使用分层评分，避免高产高切换解长期支配经验池。核心逻辑如下：

```text
若解不完整或非法：score = -inf
若 weight >= target_weight 且 setup <= target_setup + 120：进入第一档
若 weight >= target_weight - 500 且 setup <= target_setup + 120：进入第二档
若 weight >= target_weight：进入第三档
否则进入低档

score = tier + weight * 0.2 - deficit * 4.0 - setup_over * 100.0 - setup_penalty * setup
```

在本实验中，`target_weight=18500`，`target_setup=500`。这使得 `18566.21/588` 比 `18694.54/697` 更符合本轮优化目标。

## 4. 总体统计

### 4.1 候选统计

当前 `reports` 目录中共有 144 个 JSON 记录。按实际含义拆分如下：

| 类型 | 数量 | 说明 |
| --- | ---: | --- |
| 框架记录总数 | 144 | 包括正常求解摘要、重复候选、框架拒绝记录 |
| 实际进入求解/校验的候选 | 139 | 有完整 `validation_metrics` |
| 通过有限组批校验的候选 | 138 | `error_count=0` 且完整排产 |
| 求解后被校验器判为非法的候选 | 1 | `solution_iter130.json`，组批批时长不一致 |
| 未进入求解的框架级拒绝/重复记录 | 5 | 主要是重复代码、重复配置或框架拒绝 |

唯一一个求解后非法候选为：

```text
framework_summary_iter130.json
strategy_name: crossover_prune_v1
指标: 17566.51 / 532
错误: 311#氮保炉 batch duration mismatch; actual=1950, expected_max_process_time=1920
```

这说明固定校验器确实起到了安全边界作用：即使 LLM 给出某些有风险的策略，最终也不会把非法解提升为可交付解。

### 4.2 基线与最终结果

框架 seed baseline 指标为：

| 指标 | 数值 |
| --- | ---: |
| `completed_weight_within_horizon` | 18730.38 |
| `setup_count_positive` | 697 |
| `fully_scheduled_tasks` | 1751 / 1751 |
| `error_count` | 0 |

最终综合评分 best 为：

| 指标 | 数值 |
| --- | ---: |
| `completed_weight_within_horizon` | 18566.21 |
| `setup_count_positive` | 588 |
| `completed_tasks_within_horizon` | 1474 |
| `fully_scheduled_tasks` | 1751 / 1751 |
| `scheduled_ops` | 9642 |
| `batch_ops` | 538 |
| `batch_group_count` | 250 |
| `error_count` | 0 |

与 seed baseline 相比，最终 best 牺牲约 `164.17` 产量，setup 从 `697` 降到 `588`，减少 `109` 次正切换。由于本轮目标是接近 `185xx/500`，该结果比高产高切换基线更符合目标函数。

## 5. 最终推荐解与可复核命令

### 5.1 综合评分最优解

| 项目 | 内容 |
| --- | --- |
| 策略名 | `new_completion_rule_v2` |
| 候选策略 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2\candidates\deepseek_strategy_round_27_cand_00.py` |
| 解文件 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2\solutions\solution_iter120.json` |
| 指标 | `18566.21 / 588` |
| 合法性 | 有限组批校验通过，`error_count=0` |

复核命令：

```powershell
python scripts\validate_batch_solution.py `
  --input data\data1\测试算例1.json `
  --solution outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2\solutions\solution_iter120.json
```

复核输出摘要：

```json
{
  "completed_tasks_within_horizon": 1474,
  "completed_weight_within_horizon": 18566.21,
  "setup_count_positive": 588,
  "fully_scheduled_tasks": 1751,
  "valid_full_tasks": 1751,
  "total_tasks": 1751,
  "scheduled_ops": 9642,
  "batch_ops": 538,
  "batch_group_count": 250,
  "machine_count_with_nonbatch_load": 27,
  "machine_count_with_batch_load": 27,
  "error_count": 0
}
```

### 5.2 低切换折中解

| 项目 | 内容 |
| --- | --- |
| 策略名 | `rule_prune_ablation_v1` |
| 候选策略 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2\candidates\deepseek_strategy_round_27_cand_02.py` |
| 解文件 | `F:\huawei_fjsp_llm\huawei_fjsp_llm\outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2\solutions\solution_iter122.json` |
| 指标 | `18555.18 / 562` |
| 合法性 | 有限组批校验通过，`error_count=0` |
| 注意事项 | 该候选的排程解合法，但 `describe_rule_changes()` 存在规则审计自相矛盾警告 |

复核命令：

```powershell
python scripts\validate_batch_solution.py `
  --input data\data1\测试算例1.json `
  --solution outputs\deepseek_strategy_framework_case1_finite_lowsetup_diverse_v2\solutions\solution_iter122.json
```

复核输出摘要：

```json
{
  "completed_tasks_within_horizon": 1477,
  "completed_weight_within_horizon": 18555.18,
  "setup_count_positive": 562,
  "fully_scheduled_tasks": 1751,
  "valid_full_tasks": 1751,
  "total_tasks": 1751,
  "scheduled_ops": 9642,
  "batch_ops": 538,
  "batch_group_count": 241,
  "machine_count_with_nonbatch_load": 27,
  "machine_count_with_batch_load": 27,
  "error_count": 0
}
```

这里需要特别区分两个概念：`solution_iter122.json` 的排程解通过校验，是合法解；但该候选的规则说明文本中把同一项规则同时写入删除/保留，属于策略审计质量问题，不影响解文件本身的合法性。

## 6. 主要帕累托候选

下表保留了本轮实验中较有代表性的非支配或近似非支配点。`warning` 表示策略规则审计文本存在问题，不表示排程解非法。

| 产量 | setup | 轮次/候选 | 策略名 | 解文件 | 备注 |
| ---: | ---: | --- | --- | --- | --- |
| 14591.36 | 496 | round00 cand02 | `elite_baseline_with_custom_dispatch_and_setup_penalty` | `solution_iter02.json` | 极低 setup，但产量过低，只作为下界参考 |
| 17914.96 | 518 | round23 cand04 / round24 cand00 等 | `guarded_parameter_mutation_v5` / `new_dispatch_rule_v3` | `solution_iter100.json` / `solution_iter102.json` | setup 接近目标，但产量远低于 18500 |
| 18226.01 | 524 | round23 cand05 | `new_dispatch_rule_v2` | `solution_iter101.json` | 低 setup 阶段的较好恢复点 |
| 18555.18 | 562 | round27 cand02 | `rule_prune_ablation_v1` | `solution_iter122.json` | 低切换折中解；解合法，规则审计有 warning |
| 18566.21 | 588 | round27 cand00 | `new_completion_rule_v2` | `solution_iter120.json` | 本轮正式 best |
| 18626.38 | 601 | round18 cand00 | `new_completion_rule_v1` | `solution_iter70.json` | 高产和低 setup 的桥接点 |
| 18686.30 | 647 | round05 cand02 / round12 cand00 | `elite_baseline_with_path_density_and_phase2_urgency_v5` / `crossover_prune_v2` | `solution_iter17.json` / `solution_iter44.json` | 高产但 setup 偏高 |
| 18694.54 | 697 | round10 cand00/cand01 | `low_setup_family_lock_v2` / `machine_family_partition_v2` | `solution_iter36.json` / `solution_iter37.json` | 本轮最高产量候选，但 setup 接近 baseline |

从表中可以看出，在当前固定内核和策略接口下，`18550-18570` 产量附近的最低稳定 setup 已下降到 `562-588` 区间；若进一步把 setup 压到 `520` 左右，产量通常掉到 `17900-18200`；若追求 `18680+` 产量，setup 通常回到 `647-697`。

## 7. 分阶段演化过程

### 7.1 阶段 A：高产高切换基线形成（round00-round07）

初始阶段主要围绕高产量和默认路径/调度评分展开。代表结果包括：

| 阶段结果 | 指标 | 说明 |
| --- | --- | --- |
| round00 `phase2_completion_with_progress_and_setup_penalty` | `18551.09 / 650` | 第一轮即达到 18500+，但 setup 较高 |
| round02 `density_recovery_and_setup_balance` | `18634.32 / 682` | 提升密度奖励后产量上升，setup 同步上升 |
| round05 `elite_baseline_with_path_density_and_phase2_urgency_v5` | `18686.30 / 647` | 高产较稳定，setup 仍偏高 |

该阶段沉淀出的经验是：产量主要由密度、任务权重、已启动任务补全和二阶段紧迫度保护驱动；setup 若只靠普通惩罚项控制，很难压到 600 以下。

### 7.2 阶段 B：低 setup 探索与不稳定区间（round08-round17）

该阶段开始强化同族连续、零 setup、机器 family 分区、组批等待和参数回调。代表结果包括：

| 阶段结果 | 指标 | 说明 |
| --- | --- | --- |
| round08 `low_setup_exploration_v1` | `18046.98 / 568` | setup 明显下降，但产量跌破目标 |
| round12 `independent_design_v1` | `18241.89 / 570` | 低切换方向有所恢复，但仍不足 18500 |
| round13 `completion_rule_mutation_v1` | `18299.78 / 571` | 继续恢复，但产量仍低 |
| round17 `guarded_parameter_mutation_v1` | `18479.28 / 585` | 接近目标带，说明小幅参数回调比大幅规则改写更稳定 |

该阶段证明：过强 setup 惩罚和过强同族锁定可以压切换，但会牺牲关键任务闭环，导致大量任务错过 horizon。有效策略必须在二阶段恢复尾部补齐能力。

### 7.3 阶段 C：桥接点出现（round18）

round18 的 `new_completion_rule_v1` 取得 `18626.38 / 601`，成为从低 setup 区间回到高产区间的重要桥接点。

该候选的主要尝试包括：

- 删除部分与固定内核重叠的进度奖励；
- 引入路径剩余密度奖励；
- 引入机器负载均衡惩罚；
- 保留已启动任务奖励、同族连续、零 setup、紧迫度奖励；
- 提升密度权重以恢复产量。

最终结果表明：该方向能把产量拉回 `18600+`，但 setup 仍略高于 600。后续 round27 的 best 可以理解为在该桥接点基础上进行回调和删减后得到的目标带解。

### 7.4 阶段 D：低 setup 陷阱（round20-round25）

该阶段大量候选围绕路径规则、派工规则、删规则消融和低 setup 目标继续探索，但多次落入低产平台。

代表结果包括：

| 阶段结果 | 指标 | 说明 |
| --- | --- | --- |
| round20 `new_path_rule_v1` | `17875.80 / 537` | setup 下降，但产量明显不足 |
| round21 `crossover_prune_v2` | `17970.14 / 526` | setup 更低，产量仍低 |
| round23 `new_completion_rule_v2` | `17914.96 / 518` | setup 接近 500，但产量远离 18500 |
| round23 `new_dispatch_rule_v2` | `18226.01 / 524` | 最好的低 setup 恢复点之一 |

该阶段的主要结论是：如果策略过度追求 setup 接近 500，会倾向于等待同族连续或选择低切换动作，但会错过关键闭环任务，导致产量掉到 `17800-18200`。因此，目标 `185xx/500` 的可行空间并不是简单调高 setup 惩罚即可达到。

### 7.5 阶段 E：目标带突破（round26-round27）

round26 先恢复到 `18524.80 / 604`，round27 进一步得到两个关键解：

| 候选 | 指标 | 意义 |
| --- | --- | --- |
| round27 cand00 `new_completion_rule_v2` | `18566.21 / 588` | 本轮正式 best |
| round27 cand02 `rule_prune_ablation_v1` | `18555.18 / 562` | 更低 setup 的折中解 |
| round27 cand04 `guarded_parameter_mutation_v6` | `18566.21 / 588` | 与 cand00 同指标的重复优秀点 |

本阶段的核心突破是：在 `score_phase2` 中使用“密度-效率平衡因子”：

```text
density * 74.0 - setup_time * 4.0
```

这比单纯密度奖励或单纯 setup 惩罚更稳定。它使二阶段不再只补最高密度任务，也不会为降低 setup 过度牺牲产量，而是在高密度和低切换之间做局部折中。

### 7.6 阶段 F：round28-round30 的退化与饱和

最后三轮继续尝试路径时间紧迫度、机器负载均衡、路径剩余密度、删减二阶段规则、降低组批潜力等方向，但没有刷新 best。

round30 的 6 个候选结果为：

| 候选 | 指标 |
| --- | --- |
| cand0 | `17493.79 / 578` |
| cand1 | `17536.03 / 572` |
| cand2 | `17769.53 / 529` |
| cand3 | `17493.79 / 578` |
| cand4 | `17647.97 / 572` |
| cand5 | `17493.79 / 578` |

这些结果说明，在当前接口限制下，继续围绕同一批评分因子做微调已经出现饱和：过度删规则或引入路径/负载类规则容易导致产量坍塌，而回到 best 附近又容易重复 `18566/588` 或 `18555/562` 的解域。

## 8. 逐轮摘要表

以下表格按有实际求解摘要的轮次列出。轮次编号来自文件名；由于中途续跑、重复候选和框架拒绝，round01、round06、round16、round19 没有可采纳的求解摘要。

| 轮次 | 候选数 | 本轮最高产量点 | 本轮低切换代表点 | 主要观察 |
| ---: | ---: | --- | --- | --- |
| 00 | 5 | `18551.09/650` `phase2_completion_with_progress_and_setup_penalty` | `18551.09/650` | 初始高产但 setup 偏高 |
| 02 | 4 | `18634.32/682` `density_recovery_and_setup_balance` | `18551.09/650` | 密度增强提高产量，setup 上升 |
| 03 | 1 | `18551.09/650` | `18551.09/650` | 延续早期基线 |
| 04 | 2 | `18686.30/648` | `18686.30/648` | 高产高切换结构形成 |
| 05 | 6 | `18686.30/647` | `18686.30/647` | 高产稳定，setup 仍高 |
| 07 | 3 | `18686.30/648` | `18686.30/648` | 与 round05 接近 |
| 08 | 6 | `18223.53/609` | `18046.98/568` | 开始低 setup 探索，产量下降 |
| 09 | 6 | `18372.39/667` | `18372.39/667` | setup 反弹，产量未充分恢复 |
| 10 | 2 | `18694.54/697` | `18694.54/697` | 本轮最高产量，但 setup 接近 baseline |
| 11 | 6 | `18565.79/664` | `18565.79/664` | setup 仍偏高 |
| 12 | 6 | `18686.30/647` | `18241.89/570` | 同轮出现高产点和低切换恢复点 |
| 13 | 6 | `18419.31/582` | `18299.78/571` | 进入目标带附近但产量不足 |
| 14 | 6 | `18299.78/580` | `18299.78/580` | 参数变动未带来提升 |
| 15 | 2 | `18155.20/570` | `18155.20/570` | 低 setup 但产量下滑 |
| 17 | 6 | `18479.28/585` | `18155.20/568` | 参数回调接近目标带 |
| 18 | 6 | `18626.38/601` | `17946.27/536` | 桥接点出现 |
| 20 | 6 | `17875.80/537` | `17875.80/537` | 路径规则方向产量坍塌 |
| 21 | 6 | `17970.14/526` | `17914.96/522` | setup 接近目标但产量不足 |
| 22 | 6 | `17914.96/522` | `17914.96/522` | 低 setup 平台 |
| 23 | 6 | `18226.01/524` | `17914.96/518` | 低 setup 下的最好恢复点之一 |
| 24 | 6 | `17970.14/527` | `17914.96/518` | 未突破低产平台 |
| 25 | 6 | `17732.37/534` | 无 17800+ 代表点 | 继续退化 |
| 26 | 6 | `18524.80/604` | `17970.14/527` | 从低产平台恢复到目标带附近 |
| 27 | 6 | `18566.21/588` | `18216.98/535` | 正式 best 出现；另有 `18555.18/562` 重要折中点 |
| 28 | 6 | `17769.53/526` | 无 17800+ 代表点 | 再次坠入低产平台 |
| 29 | 6 | `17769.53/530` | 无 17800+ 代表点 | 无刷新 |
| 30 | 6 | `17769.53/529` | 无 17800+ 代表点 | 无刷新，显示局部饱和 |

## 9. 最终 best 的规则与参数

### 9.1 最终 best 规则变化

`new_completion_rule_v2` 的关键规则如下：

新增规则：

- 在 `score_phase2` 中加入密度-效率平衡因子：

```text
density * 74.0 - setup_time * 4.0
```

删除规则：

- 删除二阶段中较强的进度奖励，避免它与固定内核默认进度处理重叠，并减少为了推进单个任务而造成的 setup 反弹。

保留规则：

- 已启动任务奖励 `phase2_started = 2200.0`；
- 二阶段同族连续奖励 `phase2_family = 780.0`；
- 零 setup 奖励；
- 紧迫度奖励；
- 上界保护规则；
- 组批潜力奖励；
- 有限组批等待窗口 `batch_group_wait = 320`；
- 适度 setup 惩罚，而非极端 setup 惩罚。

### 9.2 最终 best 主要参数

```json
{
  "lookahead": 85,
  "start_guard": 120,
  "score_weight": 340.0,
  "score_density": 22300.0,
  "score_started": 140.0,
  "score_family": 680.0,
  "score_progress": 120.0,
  "score_zero_setup": 480.0,
  "score_setup_fixed": 480.0,
  "score_setup_per": 4.0,
  "phase2_started": 2200.0,
  "phase2_density": 6750.0,
  "phase2_family": 780.0,
  "phase2_zero_setup": 380.0,
  "phase2_setup_fixed": 580.0,
  "phase2_setup_per": 4.0,
  "phase2_allow_unstarted": true,
  "finite_batch_capacity": true,
  "batch_group_wait": 320,
  "batch_group_mixed_time": true,
  "batch_group_any_time": false
}
```

该参数组合的特点是：第一阶段保持一定产量导向，第二阶段强化已启动任务补全、同族连续和 setup 控制，同时保留 `phase2_allow_unstarted=true`，避免只补已启动任务导致部分任务永远无法完整排完。

## 10. 有效经验总结

### 10.1 有效规则

本轮实验中最稳定的有效规则包括：

- 二阶段密度-效率平衡比单独密度奖励更稳定；
- `score_path` 返回 `None`、使用固定内核默认路径规则，通常比 DeepSeek 自定义路径规则更可靠；
- `score_phase1` 大幅自定义不一定有效，保守使用内核默认第一阶段评分加参数微调更稳定；
- `phase2_started=2200` 对完整解补齐非常重要；
- 同族连续奖励和零 setup 奖励必须保留，但不能强到压制产量；
- `batch_group_wait=320`、`batch_group_mixed_time=true` 是本算例有限组批下较稳定的组合；
- `batch_group_any_time=false` 应保持不变，因为放开任意时间组批会显著增加尾部补齐和批时长风险；
- `score_density≈22300`、`phase2_density≈6750` 是目标带附近较稳定的密度区间；
- setup 惩罚系数 `4.0` 比 `4.5` 或更高值更稳，后者容易把产量压低。

### 10.2 有效参数区间

| 参数组 | 有效区间或取值 | 说明 |
| --- | --- | --- |
| `lookahead` | `85` | 比较稳定；给候选排序足够空间，同时不至于过宽 |
| `start_guard` | `120` | 保留早期启动保护 |
| `score_density` | `22250-22500`，最终 `22300` | 高于该区间易 setup 反弹，低于该区间产量不足 |
| `phase2_density` | `6700-6800`，最终 `6750` | 与 `score_density` 配合 |
| `score_family` | `680` 左右 | 过低无法控 setup，过高会牺牲产量 |
| `phase2_family` | `780` 左右 | 二阶段同族连续的核心参数 |
| `score_setup_fixed` | `480` 左右 | 过强会低产，过弱 setup 反弹 |
| `phase2_setup_fixed` | `580` 左右 | 二阶段控制切换的核心参数 |
| `score_setup_per` / `phase2_setup_per` | `4.0` | 当前最稳；`4.5+` 常导致产量坍塌 |
| `batch_group_wait` | `320` | 本算例较稳 |

### 10.3 有效搜索策略

从 30 轮结果看，有效搜索不是“每轮只调一个数字”，而是以下组合：

- 先用高产候选建立产量上限；
- 再用同族、零 setup、setup 惩罚寻找低切换方向；
- 当低 setup 导致产量坍塌时，恢复二阶段补全和密度奖励；
- 对每个新规则做消融，如果多轮低产，则把规则删掉或回退到内核默认；
- 保留多个精英点，而不是只保留当前最高产量点；
- 对候选强制分配不同探索角色，避免同轮候选同质化。

## 11. 失败方向总结

### 11.1 自定义路径规则普遍不稳定

多轮 `new_path_rule` 尝试了路径密度、路径时间紧迫度、剩余路径时间、候选机器数等规则，但多数结果落在 `17700-18000` 产量区间。原因可能是：

- 路径选择发生在调度前，缺少后续机器动态占用信息；
- 默认路径规则已经包含较强的工艺时间和候选设备偏好；
- 自定义路径规则容易把任务推向局部看似高密度、全局却拥堵的路径；
- 一旦路径选择失误，后续动作排序很难完全弥补。

因此，本轮最终经验是：在没有更强状态特征前，`score_path()` 返回 `None` 使用默认路径规则更可靠。

### 11.2 过强 setup 惩罚会进入低产平台

当 `score_setup_per` 或 `phase2_setup_per` 提高到 `4.2-4.5+`，或者同时提高同族/零 setup 奖励时，候选经常出现：

- setup 降到 `520-540`；
- 产量掉到 `17700-18200`；
- 任务闭环不足，迟完任务增加。

这说明 setup 不是可以单调压低的目标。若为降低切换等待同族连续，会牺牲关键任务的及时完工。

### 11.3 机器负载均衡惩罚未稳定生效

多轮候选尝试了机器负载均衡惩罚，例如对高负载机器额外扣分。但结果不稳定，常导致产量下降。原因可能是：

- 本问题的瓶颈设备本来就必须承担更多关键工序；
- 简单按已排工时惩罚高负载机器，会误伤真实瓶颈；
- 固定内核已有可行窗口和机器时间轴约束，额外负载惩罚容易与真实最早完工目标冲突。

因此该规则目前不建议保留，除非后续引入更细的机器瓶颈识别特征。

### 11.4 删除二阶段紧迫度/补全规则会导致尾部失控

round28-round30 多个候选尝试删除或削弱二阶段紧迫度、路径时间紧迫度、进度补全等规则，结果反复落在 `17493-17769` 产量区间。说明：

- 第二阶段不是单纯降低 setup 的阶段；
- 它还承担“把完整解补齐并尽量让尾部任务入窗”的功能；
- 如果削弱补全能力，setup 可能下降，但产量损失更大。

### 11.5 低 setup 极端点不可直接作为交付目标

`14591.36/496` 说明理论上可以把 setup 压到 500 以下，但产量严重不足。该点的价值是证明“低 setup 可达”，而不是作为可交付策略。对本算例而言，当前接口下更现实的低切换高产区间是 `18555/562` 到 `18566/588`。

## 12. 为什么没有继续超过当前 best

从最后几轮现象看，当前框架在“只允许修改路径/阶段评分函数和参数”的接口下已经接近局部饱和：

- 继续提高密度，产量可能小幅上升，但 setup 会回到 `600+`；
- 继续提高 setup 惩罚，setup 可以下降，但产量会跌破 `18200`；
- 自定义路径规则多数会破坏默认路径的稳定性；
- 机器负载、路径剩余密度、尾部紧迫度等规则缺少足够丰富的状态特征，容易过拟合；
- 多个优秀点已经在 `18555-18566` 和 `562-588` 区间重复出现，说明局部策略空间被反复搜索到。

因此，如果后续要继续突破 `185xx/500`，更可能需要开放更高层的决策接口，而不是继续只调当前几个评分项。例如：

- 允许 LLM 设计候选池分层策略，而不只是对同一个候选池排序；
- 允许 LLM 定义分阶段策略切换条件，例如早期高产、局部同族锁定、尾部强补全；
- 引入更细粒度的机器瓶颈状态、族切换状态和任务链闭环状态；
- 在固定校验器不变的前提下，开放局部重排或局部修复算子；
- 将 LLM 的规则建议与自动参数搜索结合，形成“规则生成器 + 参数调优器 + 校验器”的闭环。

## 13. 与“自我迭代框架”目标的关系

本轮实验验证了一个重要方向：即使不向 DeepSeek 暴露现有完整求解器，只给它问题文档、输入输出说明、RL 思路文档、固定策略接口和校验反馈，它也能在有限组批口径下形成有效策略搜索过程。

已经实现的能力包括：

- 自动生成候选策略代码；
- 自动运行固定内核求解；
- 自动调用校验器；
- 自动根据指标、错误和上一轮经验写下一轮 prompt；
- 自动区分候选探索角色；
- 自动记录规则新增、删除、保留和参数变化；
- 自动拒绝重复候选或低质量规则审计；
- 自动保存全部候选、解、报告和经验日志。

尚未充分实现的能力包括：

- 真正开放“如何选择动作”的算法结构，而不是只开放评分项；
- 更强的跨轮归因分析，例如自动定位哪些机器、哪些任务链导致产量损失；
- 更可靠的规则交叉与消融，避免最后几轮反复坠入同一低产平台；
- 对候选代码进行更深的语义去重，而不仅是配置签名和代码签名；
- 让 LLM 自动提出并实现新的局部搜索算子或策略切换机制。

## 14. 后续建议

若继续推进该方向，建议分三条线：

### 14.1 保留当前 best 与低 setup 折中解

建议至少保留以下两个可交付解：

- `solution_iter120.json`：`18566.21 / 588`，正式 best；
- `solution_iter122.json`：`18555.18 / 562`，低 setup 折中解。

这两个解均已通过有限组批校验，可作为后续报告和对比实验的代表点。

### 14.2 当前接口下的小幅继续搜索

如果仍在当前接口内继续，可以重点围绕：

- `score_density = 22250-22400`；
- `phase2_density = 6700-6775`；
- `phase2_family = 760-800`；
- `phase2_setup_fixed = 560-600`；
- `score_phase2` 中 `density * 74.0 - setup_time * 4.0` 的系数微调；
- `batch_group_wait = 300-340`。

但预期收益有限，且容易重复已有点。

### 14.3 下一代框架改造

更值得做的是在固定校验器不变的前提下扩大 LLM 可设计空间：

- 第一级：继续固定硬约束过滤，保证不合法动作不进入策略；
- 第二级：开放候选池构造方式，例如是否只看最早窗口、是否按机器族分层、是否按任务链闭环分层；
- 第三级：开放阶段切换逻辑，例如第一阶段何时停止、第二阶段何时允许未启动任务、何时强制补齐；
- 第四级：开放局部修复算子，例如对少量尾部任务做路径替换、同族块重排或组批拆分；
- 第五级：仍由人工校验器裁决最终合法性和指标，LLM 只根据反馈继续迭代。

这样才能从“调参 + 评分函数搜索”升级为更接近 EOH、REEVO、HeurAgenix 等思想的“规则/算子/策略结构自进化”。

## 15. 本次结论

本次 30 轮 DeepSeek 自迭代的核心结论如下：

- 框架在不暴露完整求解代码的情况下，能够自动生成、评估、反思和修改策略模块；
- 固定内核和人工校验器有效保证了复杂约束的正确性；
- 当前 best 为 `18566.21 / 588`，低切换折中解为 `18555.18 / 562`，均为完整合法有限组批解；
- 相比 seed baseline `18730.38 / 697`，最终 best 明显降低 setup，但产量略有牺牲；
- 当前接口下，setup 低于 `560` 且产量保持 `18500+` 已经较困难；
- 继续提升需要开放更高层策略结构，而不是只依赖当前评分函数和参数微调。

本实验可作为后续“校验器约束下的大模型辅助工业调度算法自进化框架”的实证材料：LLM 负责提出规则、删减规则、调整参数和总结经验；固定求解内核负责动作合法性；人工校验器负责最终裁决。这种分工能够在减少人工干预的同时，避免 LLM 直接生成复杂约束求解器所带来的合法性风险。

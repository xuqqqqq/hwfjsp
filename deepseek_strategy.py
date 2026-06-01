"""
策略模块：guarded_parameter_mutation_v1 - 基于当前最佳基线的参数微调

核心思路：
1. 当前最佳基线（new_completion_rule_v2: 18566.21/588）表现优秀，产量高且setup适中。
2. 上一轮（crossover_prune_v4: 17647.97/572）产量大幅下降，主要原因是：
   - 密度回调至22200.0，导致产量损失。
   - 机器负载均衡惩罚（-300.0）过度惩罚高负载机器，导致任务集中在低负载机器上。
   - 删除了score_phase2中的进度奖励（80.0），导致尾部补齐能力下降。
3. 本候选作为guarded_parameter_mutation角色，主要调get_config，同时保留当前最佳基线的规则：
   - 恢复score_density=22300.0（当前最佳基线值），提升产量潜力。
   - 恢复phase2_density=6750.0（当前最佳基线值），与score_density保持一致。
   - 恢复score_phase2中的进度奖励（80.0），恢复尾部补齐能力。
   - 删除机器负载均衡惩罚（-300.0），避免过度惩罚高负载机器。
   - 删除score_path中的路径时间紧迫度奖励，避免与固定内核默认路径规则重叠。
   - 小幅变异：score_phase2中组批潜力奖励从200.0降低至180.0，观察是否能进一步降低setup。
4. 预期效果：产量恢复至18500-18600，setup稳定在585-595，综合评分提升至92000+。
"""
from __future__ import annotations


def describe_strategy() -> str:
    """返回策略说明，便于实验日志记录。"""
    return "guarded_parameter_mutation_v1"


def get_config(base_config: dict, case_context: dict) -> dict:
    """返回可覆盖的求解参数。

    基于当前最佳基线参数微调：
    1. score_density=22300.0（恢复至当前最佳基线值）
    2. phase2_density=6750.0（恢复至当前最佳基线值）
    3. 保持batch_group_wait=320（已验证有效）
    4. 保持setup参数：score_setup_fixed=480.0, phase2_setup_fixed=580.0
    5. 保持同族奖励：score_family=680.0, phase2_family=780.0
    6. 保持零setup奖励：score_zero_setup=480.0, phase2_zero_setup=380.0
    7. 恢复二阶段进度奖励：phase2_progress=80.0
    8. 小幅变异：score_phase2中组批潜力奖励从200.0降低至180.0
    """
    config = dict(base_config)
    config.update(
        {
            # 保留精英基线的有效设置
            "finite_batch_capacity": True,
            "batch_group_mixed_time": True,
            "batch_group_any_time": False,
            # 组批等待时间（保持320，已验证有效）
            "batch_group_wait": 320,
            # 密度权重（恢复至当前最佳基线值22300.0）
            "score_density": 22300.0,
            "phase2_density": 6750.0,
            # setup固定惩罚（保持480，避免过度抑制）
            "score_setup_fixed": 480.0,
            "phase2_setup_fixed": 580.0,
            # 同族奖励（保持680，鼓励低切换）
            "score_family": 680.0,
            "phase2_family": 780.0,
            # 零setup奖励（保持480，鼓励低切换）
            "score_zero_setup": 480.0,
            "phase2_zero_setup": 380.0,
            # 进度激励（恢复120.0，恢复尾部补齐能力）
            "score_progress": 120.0,
            "phase2_progress": 80.0,
            # 保留精英基线的其他参数
            "lookahead": 85,
            "start_guard": 120,
            "score_weight": 340.0,
            "score_started": 140.0,
            "score_setup_per": 4.0,
            "score_est_final_per": 0.01,
            "task_bonus": {},
            "defer_task": [],
            "phase2_started": 2200.0,
            "phase2_setup_per": 4.0,
            "phase2_finish_per": 0.01,
            "phase2_allow_unstarted": True,
        }
    )
    return config


def score_path(features: dict) -> float | None:
    """路径选择评分：返回None，使用固定内核默认路径规则。

    删除规则（guarded_parameter_mutation）：
    - 删除score_path中的路径时间紧迫度奖励：与固定内核默认路径规则重叠，
      且上一轮（crossover_prune_v4）产量下降至17647.97，说明自定义路径规则
      无法有效提升产量，反而可能导致产量下降。

    保留规则（guarded_parameter_mutation）：
    - 无保留规则（本候选以参数微调为主）。
    """
    return None


def score_phase1(features: dict) -> float:
    """第一阶段评分：基于当前最佳基线参数微调。

    保留规则（从当前最佳基线吸收）：
    - 密度奖励（density * 22300.0）：恢复至当前最佳基线值
    - 同族连续奖励（680.0）：鼓励机器-family分区
    - 零setup奖励（480.0）：鼓励无切换
    - 进度奖励（120.0）：恢复尾部补齐能力
    - setup惩罚（系数4.0）：避免组批时间计算异常

    删除规则（guarded_parameter_mutation）：
    - 删除机器负载均衡惩罚（-300.0）：上一轮（crossover_prune_v4）产量下降至17647.97，
      说明该规则过度惩罚高负载机器，导致任务集中在低负载机器上，实际贡献有限。
    """
    # 提取特征
    task_weight = features.get("task_weight", 0.0)
    remaining_nonbatch_time = features.get("remaining_nonbatch_time", 1.0)
    same_family = features.get("same_family", False)
    zero_setup = features.get("zero_setup", False)
    setup_time = features.get("setup_time", 0.0)
    progress = features.get("progress", 0.0)

    # 基础分数
    score = 0.0

    # 1. 密度奖励（最高优先级，恢复至22300.0）
    if remaining_nonbatch_time > 0:
        density = task_weight / remaining_nonbatch_time
        score += density * 22300.0

    # 2. 同族连续奖励（保持680，鼓励机器-family分区）
    if same_family:
        score += 680.0

    # 3. 零setup奖励（保持480，鼓励无切换）
    if zero_setup:
        score += 480.0

    # 4. 进度奖励（保持120.0，恢复尾部补齐能力）
    score += progress * 120.0

    # 5. setup惩罚（系数4.0，避免组批时间计算异常）
    score -= setup_time * 4.0

    return score


def score_phase2(features: dict) -> float:
    """第二阶段评分：基于当前最佳基线参数微调。

    保留规则（从当前最佳基线吸收）：
    - 已启动任务奖励（2200.0）：最高优先级
    - 同族连续奖励（780.0）：鼓励机器-family分区
    - 零setup奖励（480.0）：鼓励无切换
    - 密度-效率平衡因子（density * 74.0 - setup_time * 4.0）：直接平衡密度和setup
    - 上界保护规则（阈值700，权重200）：保护接近上界的任务
    - 紧迫度奖励（阈值0.55，权重300）：保护紧迫任务
    - 进度奖励（权重80.0）：恢复尾部补齐能力
    - 组批潜力奖励（权重180.0，小幅变异）

    变异规则（guarded_parameter_mutation）：
    - 组批潜力奖励从200.0降低至180.0：观察是否能进一步降低setup至590以下，
      同时保持产量在18500+。
    """
    # 提取特征
    task_weight = features.get("task_weight", 0.0)
    remaining_nonbatch_time = features.get("remaining_nonbatch_time", 1.0)
    same_family = features.get("same_family", False)
    zero_setup = features.get("zero_setup", False)
    started = features.get("started", False)
    setup_time = features.get("setup_time", 0.0)
    estimated_final = features.get("estimated_final", 0.0)
    horizon = features.get("horizon", 24480.0)
    current_time = features.get("current_time", 0.0)
    upper_bound = features.get("upper_bound", 0.0)
    progress = features.get("progress", 0.0)

    # 基础分数
    score = 0.0

    # 1. 已启动任务奖励（最高优先级）
    if started:
        score += 2200.0

    # 2. 同族连续奖励（保持780，鼓励机器-family分区）
    if same_family:
        score += 780.0

    # 3. 零setup奖励（保持480，鼓励无切换）
    if zero_setup:
        score += 480.0

    # 4. 密度-效率平衡因子（从当前最佳基线吸收）
    if remaining_nonbatch_time > 0:
        density = task_weight / remaining_nonbatch_time
        score += density * 74.0 - setup_time * 4.0

    # 5. 上界保护规则（阈值700，权重200，从当前最佳基线吸收）
    # 若upper_bound接近horizon，给予奖励，保护接近上界的任务
    if horizon > 0 and upper_bound > 0:
        if horizon - upper_bound < 700:
            score += 200.0

    # 6. 紧迫度奖励（阈值0.55，权重300，从当前最佳基线吸收）
    # 若estimated_final接近horizon，给予奖励，保护紧迫任务
    if horizon > 0 and estimated_final > 0:
        urgency_ratio = estimated_final / horizon
        if urgency_ratio > 0.55:
            score += 300.0

    # 7. 进度奖励（恢复80.0，恢复尾部补齐能力）
    score += progress * 80.0

    # 8. 组批潜力奖励（小幅变异：从200.0降低至180.0）
    # 若任务可组批且等待时间小于batch_group_wait，给予奖励
    # 注意：此处使用简化版本，实际应基于组批潜力判断
    # 由于features中不直接提供组批潜力信息，使用estimated_final作为代理
    if horizon > 0 and estimated_final > 0:
        if estimated_final < horizon * 0.85:
            score += 180.0

    return score


def describe_rule_changes() -> dict:
    """返回规则变更说明。"""
    return {
        "added_rules": [
            "恢复score_phase2中的进度奖励（80.0）：恢复尾部补齐能力，提升产量（guarded_parameter_mutation）"
        ],
        "removed_rules": [
            "删除score_phase1中的机器负载均衡惩罚（-300.0）：上一轮（crossover_prune_v4）产量下降至17647.97，说明该规则过度惩罚高负载机器，导致任务集中在低负载机器上，实际贡献有限（guarded_parameter_mutation）",
            "删除score_path中的路径时间紧迫度奖励（权重300）：与固定内核默认路径规则重叠，且上一轮（crossover_prune_v4）产量下降至17647.97，说明自定义路径规则无法有效提升产量（guarded_parameter_mutation）",
            "删除score_phase2中的路径时间紧迫度奖励（权重200）：与固定内核默认路径规则重叠，且上一轮（crossover_prune_v4）产量下降至17647.97，说明自定义路径规则无法有效提升产量（guarded_parameter_mutation）"
        ],
        "kept_rules": [
            "score_path返回None：使用固定内核默认路径规则",
            "score_phase1中密度奖励（density * 22300.0，恢复至当前最佳基线值）",
            "score_phase1中同族连续奖励（680.0，从精英基线吸收）",
            "score_phase1中零setup奖励（480.0，从精英基线吸收）",
            "score_phase1中进度奖励（120.0，恢复尾部补齐能力）",
            "score_phase1中setup惩罚（系数4.0，避免组批时间计算异常）",
            "score_phase2中已启动任务奖励（2200.0，从当前最佳基线吸收）",
            "score_phase2中同族连续奖励（780.0，从当前最佳基线吸收）",
            "score_phase2中零setup奖励（480.0，从当前最佳基线吸收）",
            "score_phase2中密度-效率平衡因子（density * 74.0 - setup_time * 4.0，从当前最佳基线吸收）",
            "score_phase2中上界保护规则（阈值700，权重200，从当前最佳基线吸收）",
            "score_phase2中紧迫度奖励（阈值0.55，权重300，从当前最佳基线吸收）",
            "score_phase2中进度奖励（权重80.0，恢复尾部补齐能力）",
            "score_phase2中组批潜力奖励（权重180.0，小幅变异）",
            "get_config中当前最佳基线参数：batch_group_wait=320",
            "get_config中适度回调的setup参数：score_setup_fixed=480.0, phase2_setup_fixed=580.0"
        ],
        "changed_parameters": {
            "score_density": "从22200.0恢复至22300.0，恢复至当前最佳基线值，提升产量潜力（guarded_parameter_mutation）",
            "phase2_density": "从6700.0恢复至6750.0，与score_density保持一致，强化二阶段密度导向（guarded_parameter_mutation）",
            "score_phase2中组批潜力奖励": "从200.0降低至180.0，小幅变异，观察是否能进一步降低setup至590以下（guarded_parameter_mutation）",
            "score_phase2中进度奖励": "从0.0恢复至80.0，恢复尾部补齐能力，提升产量（guarded_parameter_mutation）",
            "score_phase1中机器负载均衡惩罚": "从-300.0删除至0.0，该规则过度惩罚高负载机器，导致任务集中在低负载机器上（guarded_parameter_mutation）",
            "score_path中路径时间紧迫度奖励": "从300.0删除至0.0，与固定内核默认路径规则重叠，且无法有效提升产量（guarded_parameter_mutation）",
            "score_phase2中路径时间紧迫度奖励": "从200.0删除至0.0，与固定内核默认路径规则重叠，且无法有效提升产量（guarded_parameter_mutation）"
        },
        "rule_suggestions": [
            "下一轮可尝试将score_density小幅提升至22400，观察产量是否进一步提升至18650+，同时setup是否稳定在595以下",
            "下一轮可尝试在score_phase2中引入路径剩余密度奖励的变体：若当前任务的路径密度高于全局平均密度，给予奖励（权重30.0），强化路径选择与任务调度的连贯性",
            "下一轮可尝试将score_phase2中组批潜力奖励的权重从180.0降低至160.0，观察是否能进一步降低setup至585以下"
        ],
        "no_rule_change_reason": "",
        "expected_effect": "产量恢复至18500-18600，setup稳定在585-595，综合评分提升至92000+，接近或超越当前最佳（new_completion_rule_v2: 18566.21/588）",
        "risk": "密度恢复可能导致setup小幅上升至595，但密度-效率平衡因子应能补偿；组批潜力奖励权重降低可能导致组批次数下降，但产量提升应能补偿；恢复进度奖励可能导致setup小幅上升，但产量提升应能补偿"
    }

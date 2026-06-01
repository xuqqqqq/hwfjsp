"""算子接口冒烟策略。

该文件只用于验证 `strategy_kernel_solver.py` 暴露的高层算子接口是否会被
实际调用。策略本身不追求最优指标，也不作为交付默认参数使用。
"""


ENABLE_SCORE_HOOKS = False


def describe_strategy() -> str:
    """返回策略用途说明，供自进化框架写入摘要。"""

    return "operator_smoke: 验证候选池、动作选择与组批扩展 hook 的最小策略"


def describe_rule_changes() -> dict:
    """提供规则审计字段，保持与自进化框架约定一致。"""

    return {
        "added_rules": [
            "第一阶段候选池保留零切换或高密度候选",
            "第二阶段动作选择优先补齐已启动任务",
            "有限组批额外成员按同族零切换与密度排序",
        ],
        "removed_rules": [],
        "kept_rules": ["固定内核硬约束过滤和默认评分兜底"],
        "changed_parameters": {
            "operator_enable_hooks": "打开算子 hook 以验证内核调用路径",
            "operator_phase1_window": "扩大第一阶段可观察候选窗口",
            "operator_phase2_window": "限制第二阶段冒烟候选窗口以控制运行时间",
        },
        "rule_suggestions": [],
        "no_rule_change_reason": "",
        "expected_effect": "验证 hook 可运行，不以指标提升为目标",
        "risk": "该策略可能牺牲部分指标，仅用于冒烟测试",
    }


def get_config(base_config: dict, case_context: dict) -> dict:
    """打开算子 hook，并使用较小候选上限保证冒烟测试稳定。"""

    config = dict(base_config)
    config["finite_batch_capacity"] = case_context.get("track") == "finite"
    config["operator_enable_hooks"] = True
    config["operator_phase1_window"] = 120
    config["operator_phase2_window"] = 240
    config["operator_max_candidates"] = 96
    config["batch_group_wait"] = 0
    config["batch_group_mixed_time"] = True
    config["batch_group_any_time"] = False
    return config


def select_phase1_candidates(candidates: list[dict], context: dict) -> list[int] | None:
    """保留零切换、同族或高密度候选，验证候选池重构接口。"""

    ranked = sorted(
        candidates,
        key=lambda item: (
            item.get("zero_setup", False),
            item.get("same_family", False),
            item.get("can_finish_within_horizon", False),
            item.get("density", 0.0),
            -item.get("start_delay_from_min", 0),
        ),
        reverse=True,
    )
    selected = [item["candidate_index"] for item in ranked[: min(48, len(ranked))]]
    return selected or None


def select_phase2_candidates(candidates: list[dict], context: dict) -> list[int] | None:
    """第二阶段优先让已启动任务和 qtime 紧迫任务进入比较。"""

    ranked = sorted(
        candidates,
        key=lambda item: (
            item.get("started", False),
            -max(item.get("q_slack", 10**9), 0),
            item.get("zero_setup", False),
            item.get("density", 0.0),
            -item.get("finish", 0),
        ),
        reverse=True,
    )
    selected = [item["candidate_index"] for item in ranked[: min(64, len(ranked))]]
    return selected or None


def choose_phase2_action(candidates: list[dict], context: dict) -> int | None:
    """直接选择第二阶段动作，验证动作选择算子接口。"""

    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda item: (
            item.get("started", False),
            item.get("zero_setup", False),
            item.get("same_family", False),
            item.get("density", 0.0),
            -item.get("finish", 0),
        ),
    )
    return best.get("candidate_index")


def score_batch_extra(anchor: dict, extra: dict, context: dict) -> float | None:
    """为有限组批额外成员排序，验证组批扩展算子接口。"""

    score = 0.0
    score += 1000.0 if extra.get("same_family", False) else 0.0
    score += 500.0 if extra.get("zero_setup", False) else 0.0
    score += extra.get("density", 0.0) * 20.0
    score -= max(extra.get("start", 0) - anchor.get("start", 0), 0) * 0.1
    return score

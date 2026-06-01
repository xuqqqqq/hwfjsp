"""默认 LLM 策略模块。

本文件是 `strategy_kernel_solver.py` 的默认策略。DeepSeek 后续只需要改
这一类小策略文件，而不直接改调度内核、校验器或输出格式。
"""
from __future__ import annotations


ENABLE_SCORE_HOOKS = False


def describe_strategy() -> str:
    """返回策略说明，便于实验日志记录。"""

    return "baseline finite-batch scoring strategy"


def get_config(base_config: dict, case_context: dict) -> dict:
    """返回可覆盖的求解参数。

    base_config 由固定内核提供，策略只覆盖少量启发式参数。未知算例不应写入
    任务 id 级别的 bonus；这类局部扰动应由后续迭代根据校验反馈生成。
    """

    config = dict(base_config)
    config.update(
        {
            "finite_batch_capacity": True,
            "batch_group_wait": 320,
            "batch_group_mixed_time": True,
            "batch_group_any_time": False,
            "operator_enable_hooks": False,
            "score_zero_setup": 0.0,
            "phase2_zero_setup": 0.0,
        }
    )
    return config


def score_path(features: dict) -> float | None:
    """路径选择评分。

    默认策略返回 None，表示继续使用固定内核的路径选择规则。LLM 后续可以
    在此处基于路径加工时长、组批工序数量、qtime 等特征改写路径偏好。
    """

    return None


def score_phase1(features: dict) -> float:
    """第一阶段评分。

    默认策略返回 None，表示使用固定内核内置的成熟评分函数，作为合法基线。
    后续 LLM 生成的策略可以返回 float 来覆盖该行为。
    """

    return None


def score_phase2(features: dict) -> float:
    """第二阶段评分；默认回退到固定内核内置评分函数。"""

    return None


def select_phase1_candidates(candidates: list[dict], context: dict) -> list[int] | None:
    """第一阶段候选池算子；默认不接管候选池。"""

    return None


def select_phase2_candidates(candidates: list[dict], context: dict) -> list[int] | None:
    """第二阶段候选池算子；默认不接管候选池。"""

    return None


def choose_phase1_action(candidates: list[dict], context: dict) -> int | None:
    """第一阶段动作选择算子；默认交给评分函数。"""

    return None


def choose_phase2_action(candidates: list[dict], context: dict) -> int | None:
    """第二阶段动作选择算子；默认交给评分函数。"""

    return None


def score_batch_extra(anchor: dict, extra: dict, context: dict) -> float | None:
    """有限组批额外成员排序算子；默认使用固定内核排序。"""

    return None

#!/usr/bin/env python3
"""算例级参数/规则推荐原型。

本脚本实现一个可训练推荐器的最小闭环：

1. 从输入 JSON 抽取算例特征。
2. 根据特征选择安全的参数与规则模板。
3. 生成临时 strategy 模块，交给固定约束内核运行。
4. 保存特征、推荐、求解摘要和可继续训练的经验记录。

当前版本使用可解释的规则推荐，后续可以把 `recommend_policy()` 替换为
监督学习、bandit 或 PPO 训练出的模型，而不改变求解器与校验器。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析推荐器命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Recommend FJSP strategy parameters from case-level features."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--track", choices=("finite", "relaxed"), default="finite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rl_policy_recommender"),
        help="保存特征、推荐策略、求解结果和经验记录的目录。",
    )
    parser.add_argument(
        "--target-setup",
        type=int,
        default=600,
        help="推荐器用于选择低切换模板的目标 setup。不是硬约束。",
    )
    parser.add_argument(
        "--target-weight",
        type=float,
        default=18500.0,
        help="推荐器用于报告期望区间的目标产量。不是硬约束。",
    )
    parser.add_argument(
        "--solver-timeout",
        type=int,
        default=600,
        help="调用固定内核的最长秒数。",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("heuristic", "supervised", "bandit", "ppo", "network"),
        default="bandit",
        help=(
            "策略选择模式。heuristic 使用固定经验规则；supervised 使用相似历史样本；"
            "bandit 使用 UCB/epsilon 探索；ppo 使用模板级 PPO-style softmax 策略；"
            "network 使用离线训练的策略网络 JSON。"
        ),
    )
    parser.add_argument(
        "--network-model",
        type=Path,
        default=Path("outputs/rl_policy_recommender/policy_network.json"),
        help="policy-mode=network 时读取的策略网络 JSON 模型。",
    )
    parser.add_argument(
        "--force-candidate-id",
        default="",
        help="直接指定候选模板 ID，用于定向实验；为空时由 policy-mode 自动选择。",
    )
    parser.add_argument(
        "--policy-state",
        type=Path,
        default=Path("outputs/rl_policy_recommender/policy_state.json"),
        help="保存跨轮策略统计、经验样本和 PPO-style logits 的状态文件。",
    )
    parser.add_argument(
        "--experience-source",
        action="append",
        type=Path,
        default=[],
        help="额外读取的 experience.jsonl、final_report.json 或目录，可重复传入。",
    )
    parser.add_argument(
        "--reward-setup-penalty",
        type=float,
        default=0.2,
        help="reward = 产量 - penalty * setup - 约束/耗时惩罚中的 setup 系数。",
    )
    parser.add_argument(
        "--reward-error-penalty",
        type=float,
        default=5000.0,
        help="每个校验错误的 reward 惩罚。",
    )
    parser.add_argument(
        "--reward-time-penalty",
        type=float,
        default=0.02,
        help="每秒运行时间的 reward 惩罚。",
    )
    parser.add_argument(
        "--bandit-explore",
        type=float,
        default=180.0,
        help="UCB 探索强度；越大越倾向尝试样本少的模板。",
    )
    parser.add_argument(
        "--bandit-epsilon",
        type=float,
        default=0.05,
        help="bandit 随机探索概率。",
    )
    parser.add_argument(
        "--ppo-temperature",
        type=float,
        default=1.0,
        help="模板级 PPO-style softmax 温度。",
    )
    parser.add_argument(
        "--ppo-lr",
        type=float,
        default=0.02,
        help="模板级 PPO-style 策略更新学习率。",
    )
    parser.add_argument(
        "--ppo-clip-advantage",
        type=float,
        default=500.0,
        help="模板级 PPO-style 更新中 advantage 的截断幅度。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260602,
        help="bandit/ppo 探索随机种子。",
    )
    parser.add_argument(
        "--recommend-only",
        action="store_true",
        help="只生成特征和策略文件，不调用求解器。",
    )
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    """把相对路径解析到项目根目录。"""

    return path.resolve() if path.is_absolute() else (root / path).resolve()


def num(value: Any, default: float = 0.0) -> float:
    """安全读取数值字段。"""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float], default: float = 0.0) -> float:
    """计算均值，空列表返回默认值。"""

    return statistics.fmean(values) if values else default


def extract_case_features(input_path: Path) -> dict[str, Any]:
    """从原始算例 JSON 抽取推荐器使用的全局特征。"""

    data = json.loads(input_path.read_text(encoding="utf-8"))
    tasks: dict[str, Any] = data.get("task", {}) or {}
    machines: dict[str, Any] = data.get("eqp", {}) or {}
    setup: dict[str, Any] = data.get("setup", {}) or {}
    transition: dict[str, Any] = data.get("transition", {}) or {}

    current_time = int(num(data.get("time", {}).get("current_time"), 0))
    horizon = int(num(data.get("config", {}).get("max_output_horizon"), 0))
    if horizon <= 0:
        horizon = int(num(data.get("time", {}).get("horizon"), 24480))

    release_times: list[float] = []
    delivery_times: list[float] = []
    weights: list[float] = []
    path_counts: list[float] = []
    proc_counts: list[float] = []
    candidate_counts: list[float] = []
    proc_times: list[float] = []
    qtime_min_count = 0
    qtime_max_count = 0
    batch_proc_count = 0
    one_machine_proc_count = 0
    total_proc_count = 0
    total_candidate_count = 0

    for task in tasks.values():
        release_times.append(num(task.get("earliest_ava_time"), current_time))
        delivery_times.append(num(task.get("task_delivery_time"), horizon))
        weights.append(num(task.get("final_product_weight"), 0.0))
        paths = task.get("process_path", {}) or {}
        path_counts.append(float(len(paths)))
        task_proc_count = 0
        for path in paths.values():
            process_list = path.get("process_list", {}) or {}
            task_proc_count = max(task_proc_count, len(process_list))
            for proc in process_list.values():
                total_proc_count += 1
                if proc.get("is_batch_type"):
                    batch_proc_count += 1
                eqp_list = proc.get("eqp_list", {}) or {}
                candidate_count = len(eqp_list)
                total_candidate_count += candidate_count
                candidate_counts.append(float(candidate_count))
                if candidate_count <= 1:
                    one_machine_proc_count += 1
                for option in eqp_list.values():
                    proc_times.append(num(option.get("process_time"), 0.0))
            for qtime in (path.get("qtime_info", {}) or {}).values():
                if qtime.get("min_process_interval") is not None:
                    qtime_min_count += 1
                if qtime.get("max_process_interval") is not None:
                    qtime_max_count += 1
        proc_counts.append(float(task_proc_count))

    setup_pair_count = 0
    positive_setup_count = 0
    setup_values: list[float] = []
    for row in setup.values():
        if not isinstance(row, dict):
            continue
        for value in row.values():
            setup_pair_count += 1
            setup_time = num(value, 0.0)
            if setup_time > 0:
                positive_setup_count += 1
                setup_values.append(setup_time)

    maintenance_window_count = 0
    maintenance_minutes_in_horizon = 0.0
    factory_names: set[str] = set()
    for machine in machines.values():
        factory = machine.get("factory_info")
        if factory:
            factory_names.add(str(factory))
        for start, end in machine.get("eqp_down_interval", []) or []:
            maintenance_window_count += 1
            clipped_start = max(current_time, int(num(start)))
            clipped_end = min(horizon, int(num(end)))
            if clipped_end > clipped_start:
                maintenance_minutes_in_horizon += clipped_end - clipped_start

    transition_values: list[float] = []
    for row in transition.values():
        if isinstance(row, dict):
            transition_values.extend(num(value, 0.0) for value in row.values())

    task_count = len(tasks)
    machine_count = len(machines)
    avg_proc_count = mean(proc_counts)
    batch_ratio = batch_proc_count / max(total_proc_count, 1)
    qtime_constraint_count = qtime_min_count + qtime_max_count
    qtime_per_process = qtime_constraint_count / max(total_proc_count, 1)
    qtime_ratio = min(qtime_per_process, 1.0)
    one_machine_ratio = one_machine_proc_count / max(total_proc_count, 1)
    maintenance_ratio = maintenance_minutes_in_horizon / max(machine_count * max(horizon - current_time, 1), 1)
    due_within_horizon_ratio = sum(1 for due in delivery_times if due <= horizon) / max(task_count, 1)
    ready_ratio = sum(1 for release in release_times if release <= current_time) / max(task_count, 1)
    positive_setup_ratio = positive_setup_count / max(setup_pair_count, 1)

    return {
        "input_name": input_path.name,
        "current_time": current_time,
        "horizon": horizon,
        "task_count": task_count,
        "machine_count": machine_count,
        "factory_count": len(factory_names),
        "total_weight": round(sum(weights), 4),
        "avg_weight": round(mean(weights), 4),
        "avg_paths_per_task": round(mean(path_counts), 4),
        "avg_processes_per_task": round(avg_proc_count, 4),
        "total_process_entries": total_proc_count,
        "batch_process_ratio": round(batch_ratio, 6),
        "qtime_constraint_ratio": round(qtime_ratio, 6),
        "qtime_constraint_count": qtime_constraint_count,
        "qtime_constraints_per_process": round(qtime_per_process, 6),
        "qtime_max_count": qtime_max_count,
        "avg_candidates_per_process": round(mean(candidate_counts), 4),
        "one_machine_process_ratio": round(one_machine_ratio, 6),
        "avg_process_time": round(mean(proc_times), 4),
        "ready_task_ratio": round(ready_ratio, 6),
        "due_within_horizon_ratio": round(due_within_horizon_ratio, 6),
        "maintenance_window_count": maintenance_window_count,
        "maintenance_ratio": round(maintenance_ratio, 6),
        "setup_row_count": len(setup),
        "setup_pair_count": setup_pair_count,
        "positive_setup_count": positive_setup_count,
        "positive_setup_ratio": round(positive_setup_ratio, 6),
        "avg_positive_setup_time": round(mean(setup_values), 4),
        "transition_edge_count": len(transition_values),
        "avg_transition_time": round(mean(transition_values), 4),
        "max_transition_time": max(transition_values) if transition_values else 0.0,
        "total_candidate_count": total_candidate_count,
    }


def heuristic_recommendation(features: dict[str, Any], track: str, target_setup: int) -> dict[str, Any]:
    """根据算例特征推荐安全参数和规则模板。"""

    task_count = int(features["task_count"])
    batch_ratio = float(features["batch_process_ratio"])
    one_machine_ratio = float(features["one_machine_process_ratio"])
    qtime_ratio = float(features.get("qtime_constraints_per_process", features["qtime_constraint_ratio"]))
    maintenance_ratio = float(features["maintenance_ratio"])
    setup_density = float(features.get("positive_setup_ratio", 0.0))

    if task_count >= 1000:
        size_class = "large"
    elif task_count >= 300:
        size_class = "medium"
    else:
        size_class = "small"

    # 推荐器的第一版坚持“受限方案”：只调整参数，不接管动作选择。
    config: dict[str, Any] = {
        "lookahead": 85,
        "start_guard": 120,
        "score_weight": 340.0,
        "score_density": 23500.0,
        "score_started": 140.0,
        "score_family": 400.0,
        "score_progress": 0.0,
        "score_zero_setup": 0.0,
        "score_setup_fixed": 390.0,
        "score_setup_per": 4.0,
        "score_est_final_per": 0.01,
        "phase2_started": 2200.0,
        "phase2_density": 6900.0,
        "phase2_family": 560.0,
        "phase2_progress": 0.0,
        "phase2_zero_setup": 0.0,
        "phase2_setup_fixed": 500.0,
        "phase2_setup_per": 4.0,
        "phase2_finish_per": 0.01,
        "phase2_allow_unstarted": True,
        "operator_enable_hooks": False,
    }

    reasons: list[str] = [
        "使用固定约束内核和默认成熟评分，避免 LLM/RL 直接接管动作导致产量震荡。",
    ]

    if size_class == "large":
        config["lookahead"] = 85
        reasons.append("大规模算例保持中等 lookahead，兼顾候选质量和运行时间。")
    elif size_class == "medium":
        config["lookahead"] = 75
        reasons.append("中规模算例略缩小 lookahead，降低无效远期候选。")
    else:
        config["lookahead"] = 65
        reasons.append("小规模算例缩小 lookahead，减少调度抖动。")

    if qtime_ratio > 0.35:
        config["start_guard"] = 90
        reasons.append("qtime 约束较密，减小 start_guard，避免过早推迟临界任务。")

    if one_machine_ratio > 0.6:
        config["score_density"] = 24500.0
        config["phase2_density"] = 7200.0
        reasons.append("单候选工序比例高，强化密度项以提升有限机器利用率。")

    if maintenance_ratio > 0.03:
        config["score_est_final_per"] = 0.02
        config["phase2_finish_per"] = 0.02
        reasons.append("维修占比较高，增强预计完工时间惩罚以绕开维修尾部。")

    if setup_density > 0.4 and target_setup <= 600:
        config["score_family"] = 520.0
        config["phase2_family"] = 680.0
        config["score_setup_fixed"] = 430.0
        reasons.append("setup 矩阵较密且目标偏低，温和提高同族奖励和固定切换惩罚。")

    if track == "finite":
        config["finite_batch_capacity"] = True
        config["batch_group_mixed_time"] = True
        config["batch_group_any_time"] = False
        if batch_ratio >= 0.04:
            config["batch_group_wait"] = 320
            reasons.append("组批工序占比较高，采用已验证较稳的 mixed-time same-family 组批等待。")
        elif batch_ratio > 0.0:
            config["batch_group_wait"] = 180
            reasons.append("组批工序占比较低，使用较短等待避免尾部过度合批。")
        else:
            config["batch_group_wait"] = 0
            config["batch_group_mixed_time"] = False
            reasons.append("未检测到组批工序，关闭组批等待。")
    else:
        config["finite_batch_capacity"] = False
        config["batch_group_wait"] = 0
        config["batch_group_mixed_time"] = False
        config["batch_group_any_time"] = False
        reasons.append("relaxed 轨道按无限组批产能口径运行。")

    return {
        "policy_name": f"{track}_{size_class}_safe_kernel",
        "candidate_id": "balanced_safe",
        "policy_mode": "heuristic",
        "rule_template": "safe_kernel_parameter_policy",
        "config": config,
        "reasons": reasons,
        "feature_signals": {
            "size_class": size_class,
            "batch_ratio": batch_ratio,
            "qtime_ratio": qtime_ratio,
            "one_machine_ratio": one_machine_ratio,
            "maintenance_ratio": maintenance_ratio,
            "positive_setup_ratio": round(setup_density, 6),
        },
    }


def clamp_int(value: int, low: int, high: int) -> int:
    """把整数限制在给定范围内。"""

    return max(low, min(high, int(value)))


def candidate_from(
    base: dict[str, Any],
    *,
    candidate_id: str,
    label: str,
    updates: dict[str, Any],
    reasons: list[str],
    prior_reward: float,
) -> dict[str, Any]:
    """基于安全模板生成一个候选参数臂。"""

    config = dict(base["config"])
    config.update(updates)
    if "lookahead" in config:
        config["lookahead"] = clamp_int(config["lookahead"], 45, 125)
    if "start_guard" in config:
        config["start_guard"] = clamp_int(config["start_guard"], 0, 240)
    if "batch_group_wait" in config:
        config["batch_group_wait"] = clamp_int(config["batch_group_wait"], 0, 600)

    recommendation = dict(base)
    recommendation["candidate_id"] = candidate_id
    recommendation["policy_name"] = f"{base['policy_name']}_{candidate_id}"
    recommendation["policy_mode"] = "candidate"
    recommendation["candidate_label"] = label
    recommendation["candidate_prior_reward"] = prior_reward
    recommendation["config"] = config
    recommendation["reasons"] = [*base["reasons"], *reasons]
    recommendation["feature_signals"] = dict(base.get("feature_signals", {}))
    recommendation["feature_signals"]["candidate_id"] = candidate_id
    return recommendation


def candidate_policy_library(features: dict[str, Any], track: str, target_setup: int) -> list[dict[str, Any]]:
    """生成受硬约束保护的候选参数/规则模板。

    学习器只在这些模板之间选择，避免模型产生不可控动作或无效参数。
    """

    base = heuristic_recommendation(features, track, target_setup)
    base_config = base["config"]
    lookahead = int(base_config["lookahead"])
    finite = track == "finite"

    candidates = [
        candidate_from(
            base,
            candidate_id="balanced_safe",
            label="平衡模板",
            updates={},
            reasons=["作为冷启动和回退模板，优先保证完整合法与稳定产量。"],
            prior_reward=18500.0,
        ),
        candidate_from(
            base,
            candidate_id="output_push",
            label="高产量模板",
            updates={
                "lookahead": lookahead + 10,
                "score_density": 24800.0,
                "phase2_density": 7350.0,
                "score_family": 360.0,
                "phase2_family": 500.0,
                "score_setup_fixed": 360.0,
                "phase2_setup_fixed": 470.0,
                "batch_group_wait": 280 if finite else 0,
            },
            reasons=["提高密度和候选视野，降低过强同族偏置，优先争取更多任务入窗。"],
            prior_reward=18480.0,
        ),
        candidate_from(
            base,
            candidate_id="setup_guarded",
            label="低切换模板",
            updates={
                "lookahead": lookahead + 10,
                "score_family": 650.0,
                "phase2_family": 820.0,
                "score_setup_fixed": 520.0,
                "phase2_setup_fixed": 620.0,
                "batch_group_wait": 360 if finite else 0,
            },
            reasons=["强化同族连续和固定切换惩罚，适合 setup 目标更紧的场景。"],
            prior_reward=18250.0,
        ),
        candidate_from(
            base,
            candidate_id="fast_batch",
            label="短等待组批模板",
            updates={
                "lookahead": lookahead,
                "score_density": 24500.0,
                "phase2_density": 7250.0,
                "phase2_finish_per": 0.015,
                "batch_group_wait": 180 if finite else 0,
            },
            reasons=["缩短组批等待，减少尾部合批拖延，适合组批阻塞明显的算例。"],
            prior_reward=18420.0,
        ),
        candidate_from(
            base,
            candidate_id="patient_batch",
            label="长等待组批模板",
            updates={
                "lookahead": lookahead + 5,
                "score_family": 580.0,
                "phase2_family": 720.0,
                "batch_group_wait": 420 if finite else 0,
            },
            reasons=["增加 same-family 合批耐心，牺牲少量尾部速度以换取更低切换。"],
            prior_reward=18350.0,
        ),
        candidate_from(
            base,
            candidate_id="compact_fast",
            label="短视野快速模板",
            updates={
                "lookahead": lookahead - 20,
                "start_guard": 120,
                "score_density": 24200.0,
                "phase2_density": 7200.0,
                "score_family": 420.0,
                "phase2_family": 560.0,
            },
            reasons=["缩小 lookahead，降低运行时间和远期候选噪声。"],
            prior_reward=18380.0,
        ),
        candidate_from(
            base,
            candidate_id="deadline_guard",
            label="截止线保护模板",
            updates={
                "start_guard": 60,
                "score_est_final_per": 0.02,
                "phase2_finish_per": 0.025,
                "score_density": 24000.0,
                "phase2_density": 7100.0,
                "batch_group_wait": 260 if finite else 0,
            },
            reasons=["强化预计完工时间惩罚，用于释放时间和维修窗口更紧的算例。"],
            prior_reward=18440.0,
        ),
        candidate_from(
            base,
            candidate_id="low_setup_target",
            label="强低切换模板",
            updates={
                "lookahead": lookahead + 20,
                "score_density": 22800.0,
                "phase2_density": 6500.0,
                "score_family": 820.0,
                "phase2_family": 980.0,
                "score_setup_fixed": 680.0,
                "phase2_setup_fixed": 760.0,
                "batch_group_wait": 480 if finite else 0,
            },
            reasons=["当用户明确追求低 setup 时保留强约束模板，但由 bandit 决定是否值得尝试。"],
            prior_reward=18050.0,
        ),
    ]
    return candidates


FEATURE_VECTOR_KEYS = [
    "task_count",
    "machine_count",
    "factory_count",
    "total_weight",
    "avg_weight",
    "avg_paths_per_task",
    "avg_processes_per_task",
    "batch_process_ratio",
    "qtime_constraint_ratio",
    "qtime_constraints_per_process",
    "avg_candidates_per_process",
    "one_machine_process_ratio",
    "avg_process_time",
    "ready_task_ratio",
    "due_within_horizon_ratio",
    "maintenance_ratio",
    "positive_setup_ratio",
    "avg_transition_time",
]

FEATURE_SCALES = {
    "task_count": 2000.0,
    "machine_count": 100.0,
    "factory_count": 5.0,
    "total_weight": 30000.0,
    "avg_weight": 30.0,
    "avg_paths_per_task": 3.0,
    "avg_processes_per_task": 10.0,
    "batch_process_ratio": 1.0,
    "qtime_constraint_ratio": 1.0,
    "qtime_constraints_per_process": 3.0,
    "avg_candidates_per_process": 5.0,
    "one_machine_process_ratio": 1.0,
    "avg_process_time": 1000.0,
    "ready_task_ratio": 1.0,
    "due_within_horizon_ratio": 1.0,
    "maintenance_ratio": 0.2,
    "positive_setup_ratio": 1.0,
    "avg_transition_time": 1000.0,
}


def feature_vector(features: dict[str, Any]) -> list[float]:
    """把算例特征映射为有界数值向量，供相似度和上下文策略使用。"""

    vector: list[float] = []
    for key in FEATURE_VECTOR_KEYS:
        scale = FEATURE_SCALES[key]
        value = num(features.get(key), 0.0) / scale
        vector.append(max(0.0, min(3.0, value)))
    return vector


def feature_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    """计算两个算例特征向量之间的欧氏距离。"""

    left_vec = feature_vector(left)
    right_vec = feature_vector(right)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_vec, right_vec)))


def stable_hash(text: str) -> int:
    """生成跨进程稳定的整数 hash，用于可复现实验种子。"""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def load_json(path: Path, default: Any) -> Any:
    """读取 JSON，文件不存在或损坏时返回默认值。"""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_policy_state(path: Path) -> dict[str, Any]:
    """读取跨轮策略状态。"""

    state = load_json(path, {})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("version", 1)
    state.setdefault("arms", {})
    state.setdefault("experiences", [])
    state.setdefault("ppo", {"logits": {}, "baseline": 0.0, "count": 0})
    state["ppo"].setdefault("logits", {})
    state["ppo"].setdefault("baseline", 0.0)
    state["ppo"].setdefault("count", 0)
    return state


def iter_reports_from_path(path: Path) -> list[dict[str, Any]]:
    """从 JSON/JSONL/目录中读取历史推荐报告。"""

    reports: list[dict[str, Any]] = []
    if not path.exists():
        return reports
    if path.is_dir():
        for child in [path / "experience.jsonl", path / "final_report.json"]:
            reports.extend(iter_reports_from_path(child))
        return reports
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                reports.append(item)
        return reports
    if path.suffix.lower() == ".json":
        item = load_json(path, {})
        if isinstance(item, dict):
            reports.append(item)
    return reports


def load_experiences(state: dict[str, Any], sources: list[Path]) -> list[dict[str, Any]]:
    """合并状态文件和外部文件中的经验样本。"""

    experiences: list[dict[str, Any]] = []
    for item in state.get("experiences", []):
        if isinstance(item, dict):
            experiences.append(item)
    for source in sources:
        experiences.extend(iter_reports_from_path(source))
    return experiences


def metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """从求解报告中提取统一指标。"""

    run_result = report.get("run_result") or {}
    summary = run_result.get("summary") or report.get("summary") or {}
    metrics = summary.get("validation_metrics") or summary.get("solver_metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def reward_from_report(
    report: dict[str, Any],
    *,
    setup_penalty: float,
    error_penalty: float,
    time_penalty: float,
) -> dict[str, Any] | None:
    """计算学习层 reward。

    reward 只用于参数/模板选择，不改变业务目标口径。非法或不完整解会被显著惩罚。
    """

    run_result = report.get("run_result")
    if not isinstance(run_result, dict):
        return None
    summary = run_result.get("summary") or {}
    metrics = metrics_from_report(report)
    if not metrics:
        return None

    weight = float(metrics.get("completed_weight_within_horizon", 0.0) or 0.0)
    setup = float(metrics.get("setup_count_positive", 0.0) or 0.0)
    full_tasks = int(metrics.get("fully_scheduled_tasks", 0) or 0)
    total_tasks = int(metrics.get("total_tasks", 0) or 0)
    error_count = int(summary.get("error_count", metrics.get("error_count", 0)) or 0)
    elapsed = float(run_result.get("elapsed_seconds", 0.0) or 0.0)
    missing_tasks = max(0, total_tasks - full_tasks)
    returncode = int(run_result.get("returncode", 0) or 0)

    reward = weight - setup_penalty * setup - time_penalty * elapsed
    reward -= error_penalty * error_count
    reward -= 2500.0 * missing_tasks
    if returncode != 0:
        reward -= error_penalty

    return {
        "reward": round(reward, 6),
        "weight": weight,
        "setup": setup,
        "elapsed_seconds": elapsed,
        "error_count": error_count,
        "missing_tasks": missing_tasks,
        "returncode": returncode,
        "setup_penalty": setup_penalty,
        "error_penalty": error_penalty,
        "time_penalty": time_penalty,
    }


def arm_stats(state: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """读取候选模板的累计统计。"""

    arms = state.setdefault("arms", {})
    stats = arms.setdefault(
        candidate_id,
        {
            "count": 0,
            "reward_mean": 0.0,
            "best_reward": None,
            "best_metrics": {},
        },
    )
    return stats


def with_selection_info(
    recommendation: dict[str, Any],
    *,
    mode: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """给推荐结果附加策略选择过程信息。"""

    selected = dict(recommendation)
    selected["policy_mode"] = mode
    selected["selection_info"] = details
    return selected


def select_heuristic(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """选择固定安全模板。"""

    selected = next(item for item in candidates if item["candidate_id"] == "balanced_safe")
    return with_selection_info(
        selected,
        mode="heuristic",
        details={"reason": "固定选择 balanced_safe 作为人工经验基线。"},
    )


def select_forced(candidates: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    """直接选择指定候选模板。"""

    for candidate in candidates:
        if candidate["candidate_id"] == candidate_id:
            return with_selection_info(
                candidate,
                mode="forced",
                details={"reason": "命令行 force-candidate-id 指定模板。"},
            )
    selected = select_heuristic(candidates)
    selected["policy_mode"] = "forced"
    selected["selection_info"] = {
        "reason": "force-candidate-id 未命中候选库，回退 balanced_safe。",
        "requested_candidate_id": candidate_id,
        "available_candidate_ids": [item["candidate_id"] for item in candidates],
    }
    return selected


def select_supervised(
    candidates: list[dict[str, Any]],
    features: dict[str, Any],
    experiences: list[dict[str, Any]],
) -> dict[str, Any]:
    """基于相似历史样本选择候选模板。"""

    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    best_item: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for report in experiences:
        rec = report.get("recommendation", {}) or {}
        candidate_id = rec.get("candidate_id")
        if candidate_id not in candidate_by_id:
            continue
        reward_details = report.get("reward_details") or reward_from_report(
            report,
            setup_penalty=0.2,
            error_penalty=5000.0,
            time_penalty=0.02,
        )
        if not reward_details:
            continue
        report_features = report.get("features")
        if not isinstance(report_features, dict):
            continue
        distance = feature_distance(features, report_features)
        adjusted = float(reward_details["reward"]) - 220.0 * distance
        if best_item is None or adjusted > best_item[0]:
            best_item = (adjusted, report, reward_details)

    if best_item is None:
        selected = select_heuristic(candidates)
        selected["policy_mode"] = "supervised"
        selected["selection_info"] = {
            "reason": "无可用相似经验，回退 balanced_safe。",
            "fallback": "heuristic",
        }
        return selected

    _score, best_report, reward_details = best_item
    candidate_id = best_report["recommendation"]["candidate_id"]
    selected = candidate_by_id[candidate_id]
    return with_selection_info(
        selected,
        mode="supervised",
        details={
            "reason": "选择相似历史样本中 adjusted reward 最高的模板。",
            "matched_candidate_id": candidate_id,
            "matched_input": best_report.get("input"),
            "matched_reward": reward_details["reward"],
            "matched_metrics": {
                "weight": reward_details["weight"],
                "setup": reward_details["setup"],
                "error_count": reward_details["error_count"],
            },
        },
    )


def select_bandit(
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    explore: float,
    epsilon: float,
    rng: random.Random,
) -> dict[str, Any]:
    """使用 UCB/epsilon-greedy 在候选模板之间选择。"""

    total_count = sum(int(arm_stats(state, item["candidate_id"]).get("count", 0)) for item in candidates)
    if total_count == 0:
        selected = next(item for item in candidates if item["candidate_id"] == "balanced_safe")
        return with_selection_info(
            selected,
            mode="bandit",
            details={"reason": "冷启动阶段先选择 balanced_safe 获取稳定基线。"},
        )

    untried = [
        item
        for item in candidates
        if int(arm_stats(state, item["candidate_id"]).get("count", 0)) == 0
    ]
    if untried:
        selected = max(untried, key=lambda item: float(item.get("candidate_prior_reward", 0.0)))
        return with_selection_info(
            selected,
            mode="bandit",
            details={
                "reason": "初始 sweep 阶段优先尝试尚未评估的安全模板。",
                "selected_prior_reward": selected.get("candidate_prior_reward"),
                "remaining_untried": [item["candidate_id"] for item in untried],
                "total_count": total_count,
            },
        )

    if rng.random() < epsilon:
        selected = rng.choice(candidates)
        return with_selection_info(
            selected,
            mode="bandit",
            details={
                "reason": "epsilon 随机探索。",
                "epsilon": epsilon,
                "total_count": total_count,
            },
        )

    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        stats = arm_stats(state, candidate["candidate_id"])
        count = int(stats.get("count", 0))
        mean_reward = (
            float(stats.get("reward_mean", 0.0))
            if count > 0
            else float(candidate.get("candidate_prior_reward", 0.0))
        )
        bonus = explore * math.sqrt(math.log(total_count + 1.0) / (count + 1.0))
        scored.append((mean_reward + bonus, candidate, {"count": count, "mean_reward": mean_reward, "ucb_bonus": bonus}))

    scored.sort(key=lambda item: item[0], reverse=True)
    score, selected, stats_info = scored[0]
    return with_selection_info(
        selected,
        mode="bandit",
        details={
            "reason": "选择 UCB 得分最高的模板。",
            "ucb_score": round(score, 6),
            "selected_stats": stats_info,
            "total_count": total_count,
        },
    )


def softmax(logits: list[float], temperature: float) -> list[float]:
    """计算带温度的 softmax。"""

    temperature = max(temperature, 1e-6)
    scaled = [value / temperature for value in logits]
    max_logit = max(scaled)
    exp_values = [math.exp(value - max_logit) for value in scaled]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def select_ppo(
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    temperature: float,
    rng: random.Random,
) -> dict[str, Any]:
    """选择模板级 PPO-style 策略动作。

    这里的 PPO 是高层模板选择策略，不直接选择工序动作；真正的工序级 PPO
    需要更高成本的环境 rollout，本脚本先把可训练接口和经验闭环打通。
    """

    ppo = state.setdefault("ppo", {"logits": {}, "baseline": 0.0, "count": 0})
    logits_by_id = ppo.setdefault("logits", {})
    total_count = int(ppo.get("count", 0) or 0)
    if total_count == 0:
        selected = next(item for item in candidates if item["candidate_id"] == "balanced_safe")
        return with_selection_info(
            selected,
            mode="ppo",
            details={"reason": "PPO-style 冷启动先选择 balanced_safe。", "old_prob": 1.0},
        )

    ids = [item["candidate_id"] for item in candidates]
    logits = [float(logits_by_id.get(candidate_id, 0.0)) for candidate_id in ids]
    probabilities = softmax(logits, temperature)
    draw = rng.random()
    cumulative = 0.0
    selected_index = len(candidates) - 1
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative:
            selected_index = index
            break
    selected = candidates[selected_index]
    return with_selection_info(
        selected,
        mode="ppo",
        details={
            "reason": "按 PPO-style softmax 策略采样模板。",
            "old_prob": round(probabilities[selected_index], 8),
            "old_logit": logits[selected_index],
            "temperature": temperature,
            "total_count": total_count,
        },
    )


def matvec(matrix: list[list[float]], vector: list[float], bias: list[float]) -> list[float]:
    """计算 y = matrix * vector + bias。"""

    output: list[float] = []
    for row, b_value in zip(matrix, bias):
        output.append(sum(weight * value for weight, value in zip(row, vector)) + b_value)
    return output


def relu(vector: list[float]) -> list[float]:
    """ReLU 激活函数。"""

    return [max(0.0, value) for value in vector]


def select_network(
    candidates: list[dict[str, Any]],
    features: dict[str, Any],
    *,
    model_path: Path,
) -> dict[str, Any]:
    """使用离线训练的策略网络选择候选模板。"""

    model = load_json(model_path, {})
    if not isinstance(model, dict) or not model.get("candidate_ids"):
        selected = select_heuristic(candidates)
        selected["policy_mode"] = "network"
        selected["selection_info"] = {
            "reason": "策略网络模型不存在或格式无效，回退 balanced_safe。",
            "fallback": "heuristic",
            "model_path": str(model_path),
        }
        return selected

    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    candidate_ids = [candidate_id for candidate_id in model["candidate_ids"] if candidate_id in candidate_by_id]
    if not candidate_ids:
        selected = select_heuristic(candidates)
        selected["policy_mode"] = "network"
        selected["selection_info"] = {
            "reason": "模型候选集合与当前候选库不匹配，回退 balanced_safe。",
            "fallback": "heuristic",
            "model_path": str(model_path),
        }
        return selected

    keys = model.get("feature_keys", FEATURE_VECTOR_KEYS)
    scales = model.get("feature_scales", FEATURE_SCALES)
    vector: list[float] = []
    for key in keys:
        scale = float(scales.get(key, 1.0) or 1.0)
        vector.append(max(0.0, min(3.0, num(features.get(key), 0.0) / scale)))

    weights = model.get("weights", {})
    hidden = relu(matvec(weights["w1"], vector, weights["b1"]))
    scores = matvec(weights["w2"], hidden, weights["b2"])
    model_candidate_ids = list(model["candidate_ids"])
    score_by_id = {
        candidate_id: float(scores[index])
        for index, candidate_id in enumerate(model_candidate_ids)
        if index < len(scores) and candidate_id in candidate_by_id
    }
    selected_id = max(candidate_ids, key=lambda candidate_id: score_by_id.get(candidate_id, float("-inf")))
    selected = candidate_by_id[selected_id]
    ranked = sorted(score_by_id.items(), key=lambda item: item[1], reverse=True)
    return with_selection_info(
        selected,
        mode="network",
        details={
            "reason": "使用离线训练的策略网络选择 score 最高的安全模板。",
            "model_path": str(model_path),
            "selected_score": round(score_by_id[selected_id], 6),
            "ranked_scores": [(candidate_id, round(score, 6)) for candidate_id, score in ranked],
            "training_summary": model.get("training_summary", {}),
        },
    )


def recommend_policy(
    features: dict[str, Any],
    track: str,
    target_setup: int,
    *,
    policy_mode: str = "bandit",
    policy_state: dict[str, Any] | None = None,
    experiences: list[dict[str, Any]] | None = None,
    rng: random.Random | None = None,
    bandit_explore: float = 180.0,
    bandit_epsilon: float = 0.05,
    ppo_temperature: float = 1.0,
    network_model: Path | None = None,
    force_candidate_id: str = "",
) -> dict[str, Any]:
    """根据策略模式选择参数/规则模板。"""

    candidates = candidate_policy_library(features, track, target_setup)
    state = policy_state or load_policy_state(Path("__missing_policy_state__.json"))
    history = experiences or []
    local_rng = rng or random.Random(20260602)

    if force_candidate_id:
        return select_forced(candidates, force_candidate_id)
    if policy_mode == "heuristic":
        return select_heuristic(candidates)
    if policy_mode == "supervised":
        return select_supervised(candidates, features, history)
    if policy_mode == "ppo":
        return select_ppo(candidates, state, temperature=ppo_temperature, rng=local_rng)
    if policy_mode == "network":
        return select_network(
            candidates,
            features,
            model_path=network_model or Path("outputs/rl_policy_recommender/policy_network.json"),
        )
    return select_bandit(
        candidates,
        state,
        explore=bandit_explore,
        epsilon=bandit_epsilon,
        rng=local_rng,
    )


def strategy_source(recommendation: dict[str, Any]) -> str:
    """把推荐结果写成 strategy_kernel_solver 可加载的策略模块。"""

    config_literal = repr(recommendation["config"])
    policy_name = recommendation["policy_name"]
    reasons = recommendation["reasons"]
    candidate_id = recommendation.get("candidate_id")
    policy_mode = recommendation.get("policy_mode")
    selection_info = recommendation.get("selection_info", {})
    return f'''"""自动生成的算例级推荐策略。

该文件由 scripts/rl_policy_recommender.py 生成。策略只覆盖参数，
不直接接管动作选择；硬约束仍由固定内核和校验器负责。
"""
from __future__ import annotations


ENABLE_SCORE_HOOKS = False
CONFIG = {config_literal}
REASONS = {reasons!r}


def describe_strategy() -> str:
    return {policy_name!r}


def describe_rule_changes() -> dict:
    return {{
        "added_rules": [],
        "removed_rules": [],
        "kept_rules": [
            "固定内核默认路径选择",
            "固定内核第一阶段产量优先评分",
            "固定内核第二阶段完整补齐评分",
            "有限组批仅使用 same-family 安全组批模板",
        ],
        "changed_parameters": CONFIG,
        "candidate_id": {candidate_id!r},
        "policy_mode": {policy_mode!r},
        "selection_info": {selection_info!r},
        "rule_suggestions": [
            "后续训练模型可在该安全参数基础上选择小范围变异",
            "若需要深度强化学习，应只在内核可行动作 top-K 中做 masked action",
        ],
        "no_rule_change_reason": "首版推荐器优先保证稳定性，只推荐参数和安全规则模板。",
        "expected_effect": "输出完整合法解，并作为后续监督学习/RL 的安全基线。",
        "risk": "该策略不会使用任务级 bonus，可能低于人工长期调优后的局部最优。",
        "reasons": REASONS,
    }}


def get_config(base_config: dict, case_context: dict) -> dict:
    config = dict(base_config)
    config.update(CONFIG)
    return config


def score_path(features: dict) -> float | None:
    return None


def score_phase1(features: dict) -> float | None:
    return None


def score_phase2(features: dict) -> float | None:
    return None


def select_phase1_candidates(candidates: list[dict], context: dict) -> list[int] | None:
    return None


def select_phase2_candidates(candidates: list[dict], context: dict) -> list[int] | None:
    return None


def choose_phase1_action(candidates: list[dict], context: dict) -> int | None:
    return None


def choose_phase2_action(candidates: list[dict], context: dict) -> int | None:
    return None


def score_batch_extra(anchor: dict, extra: dict, context: dict) -> float | None:
    return None
'''


def run_solver(
    *,
    root: Path,
    input_path: Path,
    track: str,
    strategy_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    """调用固定内核，返回求解摘要和子进程信息。"""

    solution_path = output_dir / "solution.json"
    summary_path = output_dir / "solver_summary.json"
    cmd = [
        sys.executable,
        str(root / "scripts" / "strategy_kernel_solver.py"),
        "--input",
        str(input_path),
        "--track",
        track,
        "--strategy",
        str(strategy_path),
        "--output",
        str(solution_path),
        "--summary",
        str(summary_path),
    ]
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.time() - started
    (output_dir / "solver_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "solver_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "solution_path": str(solution_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def save_json(path: Path, payload: Any) -> None:
    """写入格式化 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_experience(output_dir: Path, report: dict[str, Any]) -> None:
    """把本次推荐结果写成一行经验样本，供后续训练使用。"""

    experience_path = output_dir / "experience.jsonl"
    with experience_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")


def compact_experience(report: dict[str, Any], reward_details: dict[str, Any]) -> dict[str, Any]:
    """压缩经验样本，写入跨轮策略状态。"""

    recommendation = report.get("recommendation", {}) or {}
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": report.get("input"),
        "track": report.get("track"),
        "features": report.get("features"),
        "recommendation": {
            "policy_name": recommendation.get("policy_name"),
            "candidate_id": recommendation.get("candidate_id"),
            "policy_mode": recommendation.get("policy_mode"),
            "candidate_label": recommendation.get("candidate_label"),
            "selection_info": recommendation.get("selection_info"),
            "config": recommendation.get("config"),
        },
        "reward_details": reward_details,
    }


def update_arm_stats(state: dict[str, Any], candidate_id: str, reward_details: dict[str, Any]) -> None:
    """更新 UCB/bandit 使用的候选模板统计。"""

    stats = arm_stats(state, candidate_id)
    count = int(stats.get("count", 0)) + 1
    old_mean = float(stats.get("reward_mean", 0.0))
    reward = float(reward_details["reward"])
    stats["count"] = count
    stats["reward_mean"] = old_mean + (reward - old_mean) / count
    stats["last_reward"] = reward
    stats["last_metrics"] = {
        "weight": reward_details["weight"],
        "setup": reward_details["setup"],
        "error_count": reward_details["error_count"],
        "elapsed_seconds": reward_details["elapsed_seconds"],
    }
    best_reward = stats.get("best_reward")
    if best_reward is None or reward > float(best_reward):
        stats["best_reward"] = reward
        stats["best_metrics"] = dict(stats["last_metrics"])


def update_ppo_state(
    state: dict[str, Any],
    recommendation: dict[str, Any],
    reward_details: dict[str, Any],
    *,
    ppo_lr: float,
    ppo_clip_advantage: float,
) -> None:
    """更新模板级 PPO-style softmax 策略。

    该更新是高层策略近似：只调整候选模板 logits，不替代完整 PPO rollout。
    """

    if recommendation.get("policy_mode") != "ppo":
        return
    candidate_id = str(recommendation.get("candidate_id"))
    if not candidate_id:
        return
    ppo = state.setdefault("ppo", {"logits": {}, "baseline": 0.0, "count": 0})
    logits = ppo.setdefault("logits", {})
    count = int(ppo.get("count", 0) or 0) + 1
    reward = float(reward_details["reward"])
    old_baseline = float(ppo.get("baseline", 0.0) or 0.0)
    baseline = old_baseline + (reward - old_baseline) / count
    advantage = max(-ppo_clip_advantage, min(ppo_clip_advantage, reward - old_baseline))
    logits[candidate_id] = float(logits.get(candidate_id, 0.0)) + ppo_lr * advantage / max(ppo_clip_advantage, 1.0)
    ppo["count"] = count
    ppo["baseline"] = baseline
    ppo["last_update"] = {
        "candidate_id": candidate_id,
        "reward": reward,
        "old_baseline": old_baseline,
        "new_baseline": baseline,
        "clipped_advantage": advantage,
    }


def update_policy_state(
    state_path: Path,
    state: dict[str, Any],
    report: dict[str, Any],
    reward_details: dict[str, Any],
    *,
    ppo_lr: float,
    ppo_clip_advantage: float,
) -> None:
    """把本次求解反馈写回策略状态文件。"""

    recommendation = report.get("recommendation", {}) or {}
    candidate_id = str(recommendation.get("candidate_id", "unknown"))
    update_arm_stats(state, candidate_id, reward_details)
    update_ppo_state(
        state,
        recommendation,
        reward_details,
        ppo_lr=ppo_lr,
        ppo_clip_advantage=ppo_clip_advantage,
    )
    experiences = state.setdefault("experiences", [])
    experiences.append(compact_experience(report, reward_details))
    state["experiences"] = experiences[-1000:]
    state["last_reward_details"] = reward_details
    save_json(state_path, state)


def main() -> int:
    """推荐器主流程。"""

    args = parse_args()
    root = args.root.resolve()
    input_path = resolve_path(root, args.input)
    output_dir = resolve_path(root, args.output_dir)
    policy_state_path = resolve_path(root, args.policy_state)
    network_model_path = resolve_path(root, args.network_model)
    experience_sources = [resolve_path(root, path) for path in args.experience_source]
    output_dir.mkdir(parents=True, exist_ok=True)

    features = extract_case_features(input_path)
    policy_state = load_policy_state(policy_state_path)
    experiences = load_experiences(policy_state, experience_sources)
    seed_material = f"{args.seed}:{input_path}:{args.track}:{len(policy_state.get('experiences', []))}"
    rng = random.Random(args.seed + stable_hash(seed_material))
    recommendation = recommend_policy(
        features,
        args.track,
        args.target_setup,
        policy_mode=args.policy_mode,
        policy_state=policy_state,
        experiences=experiences,
        rng=rng,
        bandit_explore=args.bandit_explore,
        bandit_epsilon=args.bandit_epsilon,
        ppo_temperature=args.ppo_temperature,
        network_model=network_model_path,
        force_candidate_id=args.force_candidate_id.strip(),
    )
    strategy_path = output_dir / "recommended_strategy.py"
    strategy_path.write_text(strategy_source(recommendation), encoding="utf-8")

    save_json(output_dir / "features.json", features)
    save_json(output_dir / "recommendation.json", recommendation)

    run_result: dict[str, Any] | None = None
    reward_details: dict[str, Any] | None = None
    if not args.recommend_only:
        run_result = run_solver(
            root=root,
            input_path=input_path,
            track=args.track,
            strategy_path=strategy_path,
            output_dir=output_dir,
            timeout=args.solver_timeout,
        )
        reward_details = reward_from_report(
            {"run_result": run_result},
            setup_penalty=args.reward_setup_penalty,
            error_penalty=args.reward_error_penalty,
            time_penalty=args.reward_time_penalty,
        )

    report = {
        "input": str(input_path),
        "track": args.track,
        "policy_mode": args.policy_mode,
        "policy_state": str(policy_state_path),
        "features": features,
        "recommendation": recommendation,
        "run_result": run_result,
        "reward_details": reward_details,
    }
    if reward_details is not None:
        update_policy_state(
            policy_state_path,
            policy_state,
            report,
            reward_details,
            ppo_lr=args.ppo_lr,
            ppo_clip_advantage=args.ppo_clip_advantage,
        )
    save_json(output_dir / "final_report.json", report)
    append_experience(output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if run_result is None or run_result.get("returncode") == 0 else int(run_result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())

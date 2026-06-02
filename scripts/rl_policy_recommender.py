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
import json
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


def recommend_policy(features: dict[str, Any], track: str, target_setup: int) -> dict[str, Any]:
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


def strategy_source(recommendation: dict[str, Any]) -> str:
    """把推荐结果写成 strategy_kernel_solver 可加载的策略模块。"""

    config_literal = repr(recommendation["config"])
    policy_name = recommendation["policy_name"]
    reasons = recommendation["reasons"]
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


def main() -> int:
    """推荐器主流程。"""

    args = parse_args()
    root = args.root.resolve()
    input_path = resolve_path(root, args.input)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = extract_case_features(input_path)
    recommendation = recommend_policy(features, args.track, args.target_setup)
    strategy_path = output_dir / "recommended_strategy.py"
    strategy_path.write_text(strategy_source(recommendation), encoding="utf-8")

    save_json(output_dir / "features.json", features)
    save_json(output_dir / "recommendation.json", recommendation)

    run_result: dict[str, Any] | None = None
    if not args.recommend_only:
        run_result = run_solver(
            root=root,
            input_path=input_path,
            track=args.track,
            strategy_path=strategy_path,
            output_dir=output_dir,
            timeout=args.solver_timeout,
        )

    report = {
        "input": str(input_path),
        "track": args.track,
        "features": features,
        "recommendation": recommendation,
        "run_result": run_result,
    }
    save_json(output_dir / "final_report.json", report)
    append_experience(output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if run_result is None or run_result.get("returncode") == 0 else int(run_result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""固定约束内核 + 可替换策略模块求解入口。

本脚本把“合法性相关逻辑”和“启发式偏好逻辑”拆开：

1. 固定内核继续使用人工实现并经过校验器反复验证的解析、可行动作过滤、
   状态提交、有限组批和两阶段补齐逻辑。
2. LLM 只能生成一个小的 strategy 文件，用于覆盖少量参数、给已有路径排序、
   给可行动作打分。

因此，策略模块可以改变路径/批组/派工偏好，但不能绕过硬约束过滤，也不能
直接写解文件。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import zlib
from pathlib import Path
from types import ModuleType
from typing import Any

from rl_relaxed_solver import (
    CandidateEval,
    INF,
    RelaxedRLScheduler,
    SetupRowStore,
    build_instance,
    detect_input_json,
    dump_solution,
    infer_force_path_map_from_solution,
    parse_force_machine_map,
    parse_named_float_map,
    parse_name_set,
    register_pickle_compat_aliases,
    validate_solution,
)
from validate_batch_solution import validate_batch_solution


def to_int(value: Any) -> int:
    """把输入 JSON 中可能为字符串的数值转换为 int。"""

    return int(float(value))


DEFAULT_CONFIG: dict[str, Any] = {
    "maintenance_shift": 0,
    "path_nonbatch_mult": 3.0,
    "path_batch_weight": 1.0,
    "path_wait_weight": 1.0,
    "path_machine_penalty": {},
    "force_path": {},
    "force_machine": {},
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
    "task_bonus": {},
    "defer_task": set(),
    "phase2_started": 2200.0,
    "phase2_density": 6900.0,
    "phase2_family": 560.0,
    "phase2_progress": 0.0,
    "phase2_zero_setup": 0.0,
    "phase2_setup_fixed": 500.0,
    "phase2_setup_per": 4.0,
    "phase2_finish_per": 0.01,
    "phase2_allow_unstarted": True,
    "finite_batch_capacity": True,
    "batch_group_wait": 320,
    "batch_group_mixed_time": True,
    "batch_group_any_time": False,
}


ALLOWED_CONFIG_KEYS = set(DEFAULT_CONFIG)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Run fixed-kernel FJSP solver with a replaceable strategy module."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        type=Path,
        default=Path("strategies/deepseek_strategy.py"),
        help="策略模块路径。该模块只能提供配置和打分函数。",
    )
    parser.add_argument("--track", choices=("finite", "relaxed"), default="finite")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--instance-cache", type=Path, default=None)
    parser.add_argument("--setup-db", type=Path, default=None)
    parser.add_argument("--horizon-override", type=int, default=None)
    parser.add_argument("--rebuild-instance", action="store_true")
    parser.add_argument("--rebuild-setup", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, path: Path | None) -> Path | None:
    """把相对路径解析到项目根目录。"""

    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_strategy(path: Path) -> ModuleType:
    """从文件加载策略模块。"""

    spec = importlib.util.spec_from_file_location("llm_strategy_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import strategy module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coerce_config(raw: dict[str, Any]) -> dict[str, Any]:
    """过滤策略返回的配置，只保留内核允许的键和类型。"""

    config = dict(DEFAULT_CONFIG)
    for key, value in raw.items():
        if key not in ALLOWED_CONFIG_KEYS:
            continue
        config[key] = value
    config["path_machine_penalty"] = dict(config.get("path_machine_penalty") or {})
    config["force_path"] = dict(config.get("force_path") or {})
    config["force_machine"] = dict(config.get("force_machine") or {})
    config["task_bonus"] = dict(config.get("task_bonus") or {})
    config["defer_task"] = set(config.get("defer_task") or set())
    config["lookahead"] = int(config["lookahead"])
    config["start_guard"] = int(config["start_guard"])
    config["batch_group_wait"] = int(config["batch_group_wait"])
    return config


def default_cache_paths(root: Path, input_path: Path, track: str, strategy_path: Path) -> tuple[Path, Path]:
    """按输入、轨道和策略路径拆分缓存。"""

    raw = f"{input_path.resolve()}|{track}|{strategy_path.resolve()}".encode("utf-8")
    checksum = zlib.crc32(raw) & 0xFFFFFFFF
    return (
        root / "cache" / f"strategy_{checksum:08x}_instance.pkl",
        root / "cache" / f"strategy_{checksum:08x}_setup.sqlite",
    )


def safe_float(value: Any, fallback: float = 0.0) -> float:
    """把策略返回值安全转换成有限浮点数。"""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(result) or math.isinf(result):
        return fallback
    return result


def path_features(
    task_id: str,
    task_payload: dict[str, Any],
    path_id: str,
    path_payload: dict[str, Any],
) -> dict[str, Any]:
    """把一条候选加工路径转换为策略可读特征。

    路径选择仍然限制在输入 JSON 已给出的 `process_path` 集合内。策略只能改变
    “选择哪条既有路径”的排序，不能新增工序、机器或路径。
    """

    nonbatch_time = 0.0
    batch_time = 0.0
    min_wait = 0.0
    max_wait_count = 0
    batch_count = 0
    nonbatch_count = 0
    machines: set[str] = set()
    min_option_count = 999999
    total_option_count = 0
    process_list = path_payload.get("process_list", {})
    for proc in process_list.values():
        eqp_list = proc.get("eqp_list", {})
        option_count = len(eqp_list)
        min_option_count = min(min_option_count, option_count)
        total_option_count += option_count
        machines.update(str(machine_id) for machine_id in eqp_list)
        min_pt = min(to_int(info.get("process_time", 0)) for info in eqp_list.values())
        if proc.get("is_batch_type"):
            batch_time += min_pt
            batch_count += 1
        else:
            nonbatch_time += min_pt
            nonbatch_count += 1
    for qtime in path_payload.get("qtime_info", {}).values():
        if qtime.get("min_process_interval") is not None:
            min_wait += float(qtime["min_process_interval"])
        if qtime.get("max_process_interval") is not None:
            max_wait_count += 1
    process_count = max(len(process_list), 1)
    return {
        "task_id": task_id,
        "path_id": str(path_id),
        "task_weight": float(task_payload.get("final_product_weight", 0.0)),
        "task_priority": float(task_payload.get("task_priority", 0.0)),
        "delivery_time": task_payload.get("task_delivery_time"),
        "earliest_available_time": task_payload.get("earliest_ava_time"),
        "process_count": process_count,
        "nonbatch_count": nonbatch_count,
        "batch_count": batch_count,
        "nonbatch_time": nonbatch_time,
        "batch_time": batch_time,
        "min_wait": min_wait,
        "max_wait_count": max_wait_count,
        "machine_count": len(machines),
        "min_option_count": 0 if min_option_count == 999999 else min_option_count,
        "avg_option_count": total_option_count / process_count,
        "estimated_path_time": nonbatch_time + batch_time + min_wait,
    }


def strategy_path_overrides(
    strategy: ModuleType,
    input_path: Path,
    config: dict[str, Any],
) -> dict[str, str]:
    """调用可选 score_path()，生成 force_path 覆盖表。

    score_path() 返回 None 时表示该任务仍使用固定内核的默认路径选择规则。
    返回数值时，内核会在该任务的已有候选路径中选择分数最高的路径。
    """

    score_path = getattr(strategy, "score_path", None)
    if score_path is None:
        return dict(config.get("force_path") or {})
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    force_path = dict(config.get("force_path") or {})
    for task_id, task_payload in payload.get("task", {}).items():
        if task_id in force_path:
            continue
        best_key: tuple[float, float, str] | None = None
        best_path = ""
        for path_id, path_payload in task_payload.get("process_path", {}).items():
            features = path_features(task_id, task_payload, str(path_id), path_payload)
            try:
                raw_score = score_path(features)
            except Exception as exc:  # noqa: BLE001 - 策略异常不能中断固定内核
                print(f"[strategy-kernel] score_path failed: {exc}", file=sys.stderr, flush=True)
                raw_score = None
            if raw_score is None:
                continue
            score = safe_float(raw_score, fallback=float("-inf"))
            key = (score, -float(features["estimated_path_time"]), str(path_id))
            if best_key is None or key > best_key:
                best_key = key
                best_path = str(path_id)
        if best_path:
            force_path[task_id] = best_path
    return force_path


class StrategyScheduler(RelaxedRLScheduler):
    """只把可行动作评分委托给策略模块的调度器。"""

    def __init__(self, *args: Any, strategy_module: ModuleType, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strategy_module = strategy_module

    def candidate_features(
        self,
        eval_item: CandidateEval,
        min_start: int,
        phase: str,
    ) -> dict[str, Any]:
        """把内部 CandidateEval 转成策略可读的纯 dict 特征。"""

        task = self.instance.tasks[eval_item.task_id]
        proc = task.processes[eval_item.idx]
        remaining_nb = max(task.optimistic_nonbatch_from[eval_item.idx], 1)
        q_slack = eval_item.upper_bound - eval_item.start
        return {
            "phase": phase,
            "task_id": eval_item.task_id,
            "process_seq": proc.seq,
            "process_index": eval_item.idx,
            "process_count": len(task.processes),
            "path_id": task.path_id,
            "machine_id": eval_item.machine_id,
            "is_batch": proc.is_batch,
            "task_weight": task.weight,
            "task_priority": task.priority,
            "remaining_nonbatch_time": remaining_nb,
            "progress": eval_item.idx / max(len(task.processes) - 1, 1),
            "start": eval_item.start,
            "finish": eval_item.finish,
            "estimated_final": eval_item.est_final,
            "horizon": self.instance.horizon,
            "current_time": self.instance.current_time,
            "start_delay_from_min": eval_item.start - min_start,
            "setup_time": eval_item.setup_time,
            "same_family": eval_item.same_family,
            "zero_setup": eval_item.zero_setup,
            "started": eval_item.started,
            "option_priority": eval_item.option_priority,
            "upper_bound": eval_item.upper_bound,
            "q_slack": None if q_slack >= INF // 2 else q_slack,
        }

    def strategy_score(self, eval_item: CandidateEval, min_start: int, phase: str) -> float | None:
        """调用策略打分函数；异常时回退到底层默认评分。"""

        func_name = "score_phase1" if phase == "phase1" else "score_phase2"
        func = getattr(self.strategy_module, func_name, None)
        if func is None:
            return None
        try:
            raw_score = func(self.candidate_features(eval_item, min_start, phase))
            if raw_score is None:
                return None
            return safe_float(raw_score)
        except Exception as exc:  # noqa: BLE001 - 策略异常不能中断固定内核
            print(f"[strategy-kernel] {func_name} failed: {exc}", file=sys.stderr, flush=True)
            return None

    def score_candidate(self, eval_item: CandidateEval, min_start: int) -> tuple[float, int, int, str]:
        """第一阶段：用策略分数替代默认分数，但只作用于可行动作。"""

        score = self.strategy_score(eval_item, min_start, "phase1")
        if score is None:
            return super().score_candidate(eval_item, min_start)
        return (score, -eval_item.start, -eval_item.finish, eval_item.task_id)

    def score_candidate_phase2(self, eval_item: CandidateEval, min_start: int) -> tuple[float, int, int, str]:
        """第二阶段：用策略分数替代默认分数，但只作用于可行动作。"""

        score = self.strategy_score(eval_item, min_start, "phase2")
        if score is None:
            return super().score_candidate_phase2(eval_item, min_start)
        return (score, -eval_item.start, -eval_item.finish, eval_item.task_id)


def build_case_context(input_path: Path, track: str) -> dict[str, Any]:
    """提供给策略模块的只读算例上下文。"""

    return {
        "input_name": input_path.name,
        "input_stem": input_path.stem,
        "track": track,
    }


def strategy_config(strategy: ModuleType, input_path: Path, track: str) -> dict[str, Any]:
    """调用策略 get_config，并合并默认配置。"""

    base = dict(DEFAULT_CONFIG)
    base["finite_batch_capacity"] = track == "finite"
    get_config = getattr(strategy, "get_config", None)
    if get_config is None:
        return coerce_config(base)
    raw = get_config(dict(base), build_case_context(input_path, track))
    if raw is None:
        raw = base
    if not isinstance(raw, dict):
        raise RuntimeError("strategy get_config() must return a dict")
    return coerce_config(raw)


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    """压缩摘要中的大字段，避免 force_path 挤占 LLM 反馈上下文。"""

    summary: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, set):
            summary[key] = sorted(value)
        elif key == "force_path" and isinstance(value, dict) and len(value) > 40:
            summary[key] = {
                "count": len(value),
                "sample": dict(list(value.items())[:20]),
            }
        else:
            summary[key] = value
    return summary


def validate_output(
    *,
    track: str,
    instance: Any,
    setup_store: SetupRowStore,
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """根据轨道调用对应校验器。"""

    if track == "finite":
        force_path_map = infer_force_path_map_from_solution(output_path)
        return validate_batch_solution(
            instance=instance,
            setup_store=setup_store,
            input_path=input_path,
            solution_path=output_path,
            force_path_map=force_path_map,
            path_nonbatch_mult=float(config["path_nonbatch_mult"]),
            path_batch_weight=float(config["path_batch_weight"]),
            path_wait_weight=float(config["path_wait_weight"]),
            path_machine_penalties=config["path_machine_penalty"],
        )
    return validate_solution(instance, setup_store, output_path)


def write_summary(path: Path | None, summary: dict[str, Any]) -> None:
    """写出实验摘要。"""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """执行固定内核求解和校验。"""

    register_pickle_compat_aliases()
    args = parse_args()
    root = args.root.resolve()
    input_path = resolve_path(root, args.input) or detect_input_json(root)
    output_path = resolve_path(root, args.output)
    strategy_path = resolve_path(root, args.strategy)
    summary_path = resolve_path(root, args.summary)
    assert input_path is not None
    assert output_path is not None
    assert strategy_path is not None

    strategy = load_strategy(strategy_path)
    config = strategy_config(strategy, input_path, args.track)
    if args.track == "relaxed":
        config["finite_batch_capacity"] = False
    config["force_path"] = strategy_path_overrides(strategy, input_path, config)

    default_instance_cache, default_setup_db = default_cache_paths(
        root, input_path, args.track, strategy_path
    )
    instance_cache = resolve_path(root, args.instance_cache) or default_instance_cache
    setup_db = resolve_path(root, args.setup_db) or default_setup_db

    instance = build_instance(
        root,
        input_path,
        instance_cache,
        force=args.rebuild_instance,
        path_nonbatch_mult=float(config["path_nonbatch_mult"]),
        path_batch_weight=float(config["path_batch_weight"]),
        path_wait_weight=float(config["path_wait_weight"]),
        path_machine_penalties=config["path_machine_penalty"],
        force_path_map=config["force_path"],
        current_time_override=None,
        zero_current_time=False,
        maintenance_shift=int(config["maintenance_shift"]),
    )
    if args.horizon_override is not None:
        instance.horizon = int(args.horizon_override)

    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=args.rebuild_setup)

    try:
        scheduler = StrategyScheduler(
            instance,
            setup_store,
            lookahead=int(config["lookahead"]),
            start_guard=int(config["start_guard"]),
            score_weight=float(config["score_weight"]),
            score_density=float(config["score_density"]),
            score_started=float(config["score_started"]),
            score_family=float(config["score_family"]),
            score_progress=float(config["score_progress"]),
            score_zero_setup=float(config["score_zero_setup"]),
            score_setup_fixed=float(config["score_setup_fixed"]),
            score_setup_per=float(config["score_setup_per"]),
            score_est_final_per=float(config["score_est_final_per"]),
            task_bonus_map=config["task_bonus"],
            defer_task_ids=config["defer_task"],
            force_machine_map=config["force_machine"],
            phase2_started=float(config["phase2_started"]),
            phase2_density=float(config["phase2_density"]),
            phase2_family=float(config["phase2_family"]),
            phase2_progress=float(config["phase2_progress"]),
            phase2_zero_setup=float(config["phase2_zero_setup"]),
            phase2_setup_fixed=float(config["phase2_setup_fixed"]),
            phase2_setup_per=float(config["phase2_setup_per"]),
            phase2_finish_per=float(config["phase2_finish_per"]),
            phase2_started_gate=not bool(config["phase2_allow_unstarted"]),
            finite_batch_capacity=bool(config["finite_batch_capacity"]),
            batch_group_wait=int(config["batch_group_wait"]),
            batch_group_mixed_time=bool(config["batch_group_mixed_time"]),
            batch_group_any_time=bool(config["batch_group_any_time"]),
            strategy_module=strategy,
        )
        task_records = scheduler.solve()
        dump_solution(instance, task_records, output_path)
        solver_metrics = scheduler.metrics()
        errors, validation_metrics = validate_output(
            track=args.track,
            instance=instance,
            setup_store=setup_store,
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
    finally:
        setup_store.close()

    strategy_name = getattr(strategy, "describe_strategy", lambda: strategy_path.name)()
    try:
        rule_changes = getattr(strategy, "describe_rule_changes", lambda: {})()
    except Exception as exc:
        rule_changes = {"error": f"describe_rule_changes failed: {exc}"}
    summary = {
        "strategy": str(strategy_path),
        "strategy_name": str(strategy_name),
        "rule_changes": rule_changes,
        "track": args.track,
        "input": str(input_path),
        "output": str(output_path),
        "config": summarize_config(config),
        "solver_metrics": solver_metrics,
        "validation_metrics": validation_metrics,
        "error_count": len(errors),
        "sample_errors": errors[:40],
    }
    write_summary(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

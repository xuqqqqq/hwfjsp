#!/usr/bin/env python3
"""批量调参与自动搜索入口。

本文件将“生成参数 -> 求解 -> 校验 -> 保存结果 -> 筛选帕累托前沿”
组织为可重复执行的命令行流程。它既支持显式 --grid 网格扫描，
也支持 --auto 自适应搜索。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

from rl_relaxed_solver import (
    RelaxedRLScheduler,
    SetupRowStore,
    build_instance,
    detect_input_json,
    dump_solution,
    parse_force_machine_map,
    parse_name_set,
    parse_named_float_map,
    register_pickle_compat_aliases,
)
from solve_best_preset import (
    CASE_PRESETS,
    DEFAULT_PRESETS,
    SolverPreset,
    default_cache_paths,
    known_case_slug,
    run_validate,
)


# 下列字段可直接传给 RelaxedRLScheduler。脚本仅扫描白名单中的参数，
# 以避免非调度参数进入 --grid 后产生静默错误。
TUNABLE_FIELDS: set[str] = {
    "lookahead",
    "start_guard",
    "score_weight",
    "score_density",
    "score_started",
    "score_family",
    "score_progress",
    "score_zero_setup",
    "score_setup_fixed",
    "score_setup_per",
    "score_est_final_per",
    "phase2_started",
    "phase2_density",
    "phase2_family",
    "phase2_progress",
    "phase2_zero_setup",
    "phase2_setup_fixed",
    "phase2_setup_per",
    "phase2_finish_per",
    "phase2_allow_unstarted",
    "batch_group_wait",
    "batch_group_mixed_time",
    "batch_group_any_time",
}

INT_FIELDS = {"lookahead", "start_guard", "batch_group_wait"}
BOOL_FIELDS = {"phase2_allow_unstarted", "batch_group_mixed_time", "batch_group_any_time"}

# 自动调参的搜索边界。该范围属于工程安全范围，而非理论最优范围：
# 范围过窄会限制探索，范围过宽会增加明显劣质解的评估成本。
AUTO_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "lookahead": (30, 140),
    "start_guard": (0, 480),
    "score_weight": (80, 700),
    "score_density": (8000, 42000),
    "score_started": (0, 500),
    "score_family": (0, 1200),
    "score_progress": (0, 1000),
    "score_zero_setup": (0, 1200),
    "score_setup_fixed": (0, 1000),
    "score_setup_per": (0, 10),
    "score_est_final_per": (0, 0.1),
    "phase2_started": (0, 4200),
    "phase2_density": (2000, 16000),
    "phase2_family": (0, 1400),
    "phase2_progress": (0, 1000),
    "phase2_zero_setup": (0, 1500),
    "phase2_setup_fixed": (0, 1200),
    "phase2_setup_per": (0, 10),
    "phase2_finish_per": (0, 0.1),
    "batch_group_wait": (0, 720),
}

# 自动搜索优先动这些连续/整数参数。布尔参数单独低概率翻转，避免轨道策略
# 每一轮都剧烈变化导致指标震荡。
AUTO_MUTABLE_FIELDS: tuple[str, ...] = (
    "lookahead",
    "start_guard",
    "score_weight",
    "score_density",
    "score_started",
    "score_family",
    "score_zero_setup",
    "score_setup_fixed",
    "score_setup_per",
    "phase2_started",
    "phase2_density",
    "phase2_family",
    "phase2_zero_setup",
    "phase2_setup_fixed",
    "phase2_setup_per",
)

AUTO_FINITE_EXTRA_FIELDS: tuple[str, ...] = ("batch_group_wait",)
AUTO_BOOL_FIELDS: tuple[str, ...] = (
    "phase2_allow_unstarted",
    "batch_group_mixed_time",
    "batch_group_any_time",
)


def parse_args(
    argv: Sequence[str] | None = None,
    default_track: str = "relaxed",
) -> argparse.Namespace:
    """解析批量调参参数。

    wrapper 脚本会通过 default_track / forced_track 固定 relaxed 或 finite 轨道。
    """

    parser = argparse.ArgumentParser(
        description="Batch parameter tuning for Huawei FJSP preset solver"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--track",
        choices=sorted(DEFAULT_PRESETS),
        default=default_track,
        help="relaxed=组批无限产能；finite=组批有限产能",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch_parameter_tune"),
        help="调参输出目录",
    )
    parser.add_argument(
        "--horizon",
        "--horizon-override",
        dest="horizon_override",
        type=int,
        default=None,
        help="覆盖产量统计截止时间；不传则使用输入 JSON",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="NAME=V1,V2,...",
        help="参数取值列表，可重复传入；默认使用内置轻量网格",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="对 --grid 指定的参数做笛卡尔积；默认只做一次改一个参数的扫描",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=12,
        help="最多运行多少组参数，包含 baseline",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=0,
        help="在网格扫描后追加多少组随机扰动参数",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="启用自适应自动调参；给定运行预算后由脚本生成候选参数",
    )
    parser.add_argument(
        "--auto-round-size",
        type=int,
        default=4,
        help="自动调参每轮生成多少个候选参数",
    )
    parser.add_argument(
        "--auto-elite-size",
        type=int,
        default=5,
        help="自动调参每轮从多少个历史优秀解中选择父代",
    )
    parser.add_argument(
        "--auto-setup-penalty",
        type=float,
        default=0.2,
        help="自动调参内部排序时每多一次 setup 折算的产量惩罚",
    )
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument(
        "--task-bonus",
        type=str,
        default="",
        help="局部微调用：固定给若干 task_id 增加评分 bonus，格式 task=bonus,...",
    )
    parser.add_argument(
        "--defer-task",
        type=str,
        default="",
        help="局部微调用：固定把若干任务推迟到第二阶段，格式 task1,task2,...",
    )
    parser.add_argument(
        "--force-machine",
        type=str,
        default="",
        help="局部微调用：固定某些工序机器，格式 task_id:seq=machine_id,...",
    )
    parser.add_argument("--rebuild-instance", action="store_true")
    parser.add_argument("--rebuild-setup", action="store_true")
    return parser.parse_args(argv)


def resolve_path(root: Path, path: Path) -> Path:
    """把相对路径解析到项目根目录下。"""

    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_scalar(field: str, raw: str) -> int | float | bool:
    """按字段类型解析 --grid 中的单个值。"""

    text = raw.strip()
    if field in BOOL_FIELDS:
        if text.lower() in {"1", "true", "yes", "y", "on"}:
            return True
        if text.lower() in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"{field} expects boolean values, got {raw!r}")
    if field in INT_FIELDS:
        return int(float(text))
    return float(text)


def parse_grid_specs(grid_specs: Iterable[str]) -> dict[str, list[int | float | bool]]:
    """解析多组 --grid name=v1,v2 参数。"""

    grids: dict[str, list[int | float | bool]] = {}
    for spec in grid_specs:
        if "=" not in spec:
            raise ValueError(f"--grid must be NAME=V1,V2,..., got {spec!r}")
        name, raw_values = spec.split("=", 1)
        name = name.strip()
        if name not in TUNABLE_FIELDS:
            allowed = ", ".join(sorted(TUNABLE_FIELDS))
            raise ValueError(f"Unsupported grid parameter {name!r}; allowed: {allowed}")
        values = [parse_scalar(name, item) for item in raw_values.split(",") if item.strip()]
        if not values:
            raise ValueError(f"--grid {name} has no values")
        grids[name] = values
    return grids


def default_one_factor_grids(base: dict[str, Any], track: str) -> dict[str, list[int | float | bool]]:
    """生成默认轻量扫描空间。

    默认网格不做笛卡尔积，而是围绕 baseline 一次只调整一个参数，
    以控制新算例首次试算的求解次数。
    """

    def around_float(name: str, lo_mult: float, hi_mult: float) -> list[float]:
        value = float(base[name])
        return [round(value * lo_mult, 6), value, round(value * hi_mult, 6)]

    grids: dict[str, list[int | float | bool]] = {
        "lookahead": [max(30, int(base["lookahead"]) - 10), int(base["lookahead"]), int(base["lookahead"]) + 10],
        "start_guard": [max(0, int(base["start_guard"]) - 60), int(base["start_guard"]), int(base["start_guard"]) + 60],
        "score_density": around_float("score_density", 0.94, 1.06),
        "score_family": around_float("score_family", 0.8, 1.2),
        "score_setup_fixed": around_float("score_setup_fixed", 0.85, 1.15),
        "phase2_density": around_float("phase2_density", 0.9, 1.1),
        "phase2_family": around_float("phase2_family", 0.8, 1.2),
    }
    if track == "finite":
        wait = int(base["batch_group_wait"])
        grids["batch_group_wait"] = [max(0, wait - 30), wait, wait + 30]
    return grids


def variant_key(params: dict[str, Any]) -> str:
    """为参数组合生成短哈希，方便命名输出文件。"""

    payload = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def add_unique_variant(
    variants: list[tuple[str, dict[str, Any]]],
    seen: set[str],
    name: str,
    params: dict[str, Any],
) -> None:
    """追加去重后的参数组合。"""

    key = variant_key(params)
    if key in seen:
        return
    seen.add(key)
    variants.append((name, params))


def build_variants(
    base: dict[str, Any],
    grids: dict[str, list[int | float | bool]],
    cartesian: bool,
    max_runs: int,
    random_trials: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """构造待运行的参数组合列表。

    非 auto 模式下，所有候选在正式求解前一次性生成。
    auto 模式下，候选会在 main() 中根据历史结果逐轮生成。
    """

    variants: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    add_unique_variant(variants, seen, "baseline", dict(base))
    if len(variants) >= max_runs:
        return variants

    if cartesian:
        names = list(grids)
        for values in product(*(grids[name] for name in names)):
            params = dict(base)
            params.update(dict(zip(names, values)))
            add_unique_variant(variants, seen, "cartesian", params)
            if len(variants) >= max_runs:
                return variants
    else:
        for name, values in grids.items():
            for value in values:
                params = dict(base)
                params[name] = value
                add_unique_variant(variants, seen, f"{name}_{value}", params)
                if len(variants) >= max_runs:
                    return variants

    rng = random.Random(seed)
    for idx in range(random_trials):
        params = dict(base)
        params["lookahead"] = max(30, int(round(float(base["lookahead"]) + rng.randint(-15, 15))))
        params["start_guard"] = max(0, int(round(float(base["start_guard"]) + rng.randint(-90, 90))))
        params["score_density"] = max(1.0, float(base["score_density"]) * rng.uniform(0.9, 1.1))
        params["score_family"] = max(0.0, float(base["score_family"]) * rng.uniform(0.75, 1.25))
        params["score_setup_fixed"] = max(0.0, float(base["score_setup_fixed"]) * rng.uniform(0.8, 1.2))
        params["score_setup_per"] = max(0.0, float(base["score_setup_per"]) + rng.uniform(-0.7, 0.7))
        params["phase2_density"] = max(1.0, float(base["phase2_density"]) * rng.uniform(0.85, 1.15))
        params["phase2_family"] = max(0.0, float(base["phase2_family"]) * rng.uniform(0.75, 1.25))
        add_unique_variant(variants, seen, f"random_{idx + 1:03d}", params)
        if len(variants) >= max_runs:
            return variants
    return variants[:max_runs]


def clamp_auto_value(field: str, value: float | int | bool) -> int | float | bool:
    """把自动扰动后的参数裁剪回安全范围。"""

    if field in BOOL_FIELDS:
        return bool(value)
    lo, hi = AUTO_FIELD_BOUNDS[field]
    clipped = max(lo, min(hi, float(value)))
    if field in INT_FIELDS:
        return int(round(clipped))
    return round(clipped, 6)


def mutate_auto_field(
    rng: random.Random,
    params: dict[str, Any],
    field: str,
    radius: float,
) -> None:
    """对单个参数做一次自适应扰动。"""

    if field in BOOL_FIELDS:
        params[field] = not bool(params[field])
        return
    lo, hi = AUTO_FIELD_BOUNDS[field]
    span = hi - lo
    current = float(params[field])
    if field in INT_FIELDS:
        step = max(1, int(round(span * radius * 0.22)))
        params[field] = clamp_auto_value(field, current + rng.randint(-step, step))
        return

    # 对非零参数优先做相对扰动；对 0 参数补一个绝对步长，否则永远离不开 0。
    relative_step = abs(current) * rng.uniform(-0.45, 0.45) * radius
    absolute_step = span * rng.uniform(-0.08, 0.08) * radius
    if abs(current) < 1e-9:
        delta = absolute_step
    else:
        delta = relative_step + absolute_step * 0.35
    params[field] = clamp_auto_value(field, current + delta)


def auto_score(result: dict[str, Any], setup_penalty: float) -> float:
    """把一个已验证结果折算成自动调参内部排序分数。"""

    metrics = result["validation_metrics"]
    if not is_valid_full(metrics):
        return -1e18
    weight = float(metrics["completed_weight_within_horizon"])
    setup = int(metrics["setup_count_positive"])
    return weight - setup_penalty * setup


def auto_parent_pool(
    base: dict[str, Any],
    results: list[dict[str, Any]],
    elite_size: int,
    setup_penalty: float,
) -> list[dict[str, Any]]:
    """选择下一轮自动搜索的父代参数。"""

    valid_results = [item for item in results if is_valid_full(item["validation_metrics"])]
    if not valid_results:
        return [base]
    by_score = sorted(
        valid_results,
        key=lambda item: auto_score(item, setup_penalty),
        reverse=True,
    )
    by_weight = sorted(
        valid_results,
        key=lambda item: (
            float(item["validation_metrics"]["completed_weight_within_horizon"]),
            -int(item["validation_metrics"]["setup_count_positive"]),
        ),
        reverse=True,
    )
    by_setup = sorted(
        valid_results,
        key=lambda item: (
            int(item["validation_metrics"]["setup_count_positive"]),
            -float(item["validation_metrics"]["completed_weight_within_horizon"]),
        ),
    )

    # 同时保留“综合得分高”“产量最高”“setup 最低”的父代，避免搜索方向单一化。
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*by_score[:elite_size], *by_weight[:2], *by_setup[:2]]:
        key = variant_key(item["params"])
        if key in seen:
            continue
        seen.add(key)
        pool.append(item["params"])
    return pool or [base]


def auto_radius(completed_runs: int, max_runs: int) -> float:
    """根据已运行比例逐步缩小搜索半径。"""

    if max_runs <= 1:
        return 0.18
    progress = min(1.0, max(0.0, completed_runs / max_runs))
    return max(0.16, 0.55 * (1.0 - progress) + 0.16 * progress)


def propose_auto_variants(
    base: dict[str, Any],
    results: list[dict[str, Any]],
    track: str,
    seen: set[str],
    rng: random.Random,
    needed: int,
    completed_runs: int,
    max_runs: int,
    elite_size: int,
    setup_penalty: float,
) -> list[tuple[str, dict[str, Any]]]:
    """根据历史结果自动生成下一批参数组合。

    生成逻辑：从优秀历史参数中选父代，随机扰动 2 到 5 个字段，
    再按参数哈希去重。该机制可在无需显式枚举参数组合的情况下持续探索。
    """

    candidates: list[tuple[str, dict[str, Any]]] = []
    if needed <= 0:
        return candidates

    parents = auto_parent_pool(base, results, elite_size, setup_penalty)
    mutable_fields = list(AUTO_MUTABLE_FIELDS)
    if track == "finite":
        mutable_fields.extend(AUTO_FINITE_EXTRA_FIELDS)
    radius = auto_radius(completed_runs, max_runs)

    attempts = 0
    while len(candidates) < needed and attempts < needed * 80:
        attempts += 1
        parent = dict(rng.choice(parents))
        params = dict(parent)
        change_count = rng.randint(2, min(5, len(mutable_fields)))
        for field in rng.sample(mutable_fields, change_count):
            mutate_auto_field(rng, params, field, radius)

        # 布尔策略是“大开关”，只用低概率翻转。有限组批时允许探索 mixed/any-time。
        bool_choices = ["phase2_allow_unstarted"]
        if track == "finite":
            bool_choices.extend(["batch_group_mixed_time", "batch_group_any_time"])
        for field in bool_choices:
            if field in params and rng.random() < 0.12:
                mutate_auto_field(rng, params, field, radius)

        # 有限组批的 mixed_time 与 any_time 同时开启时语义重叠，优先保留 any_time。
        if track == "finite" and params.get("batch_group_any_time"):
            params["batch_group_mixed_time"] = False

        add_unique_variant(
            candidates,
            seen,
            f"auto_r{completed_runs + len(candidates) + 1:03d}",
            params,
        )
    return candidates


def scheduler_kwargs(params: dict[str, Any], task_bonus_map: dict[str, float], defer_task_ids: set[str], force_machine_map: dict[tuple[str, str], str]) -> dict[str, Any]:
    """把参数字典转成 RelaxedRLScheduler 构造参数。

    本函数也是参数名白名单：只有传入本函数的字段会影响求解器。
    """

    return {
        "lookahead": int(params["lookahead"]),
        "start_guard": int(params["start_guard"]),
        "score_weight": float(params["score_weight"]),
        "score_density": float(params["score_density"]),
        "score_started": float(params["score_started"]),
        "score_family": float(params["score_family"]),
        "score_progress": float(params["score_progress"]),
        "score_zero_setup": float(params["score_zero_setup"]),
        "score_setup_fixed": float(params["score_setup_fixed"]),
        "score_setup_per": float(params["score_setup_per"]),
        "score_est_final_per": float(params["score_est_final_per"]),
        "task_bonus_map": task_bonus_map,
        "defer_task_ids": defer_task_ids,
        "force_machine_map": force_machine_map,
        "phase2_started": float(params["phase2_started"]),
        "phase2_density": float(params["phase2_density"]),
        "phase2_family": float(params["phase2_family"]),
        "phase2_progress": float(params["phase2_progress"]),
        "phase2_zero_setup": float(params["phase2_zero_setup"]),
        "phase2_setup_fixed": float(params["phase2_setup_fixed"]),
        "phase2_setup_per": float(params["phase2_setup_per"]),
        "phase2_finish_per": float(params["phase2_finish_per"]),
        "phase2_started_gate": not bool(params["phase2_allow_unstarted"]),
        "finite_batch_capacity": bool(params["finite_batch_capacity"]),
        "batch_group_wait": int(params["batch_group_wait"]),
        "batch_group_mixed_time": bool(params["batch_group_mixed_time"]),
        "batch_group_any_time": bool(params["batch_group_any_time"]),
    }


def is_valid_full(metrics: dict[str, Any]) -> bool:
    """判断一个校验结果是否是完整合法解。"""

    full_count = metrics.get("valid_full_tasks", metrics.get("fully_scheduled_tasks", 0))
    return (
        metrics.get("error_count", 1) == 0
        and full_count == metrics.get("total_tasks")
    )


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断 left 是否在产量/setup 两个目标上支配 right。"""

    lm = left["validation_metrics"]
    rm = right["validation_metrics"]
    left_weight = float(lm["completed_weight_within_horizon"])
    right_weight = float(rm["completed_weight_within_horizon"])
    left_setup = int(lm["setup_count_positive"])
    right_setup = int(rm["setup_count_positive"])
    return (
        left_weight >= right_weight
        and left_setup <= right_setup
        and (left_weight > right_weight or left_setup < right_setup)
    )


def extract_frontier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从完整合法解中筛选帕累托前沿。"""

    candidates = [item for item in results if is_valid_full(item["validation_metrics"])]
    frontier = [
        item for item in candidates
        if not any(dominates(other, item) for other in candidates if other is not item)
    ]
    frontier.sort(
        key=lambda item: (
            int(item["validation_metrics"]["setup_count_positive"]),
            -float(item["validation_metrics"]["completed_weight_within_horizon"]),
            item["name"],
        )
    )
    return frontier


def write_frontier_md(root: Path, output_dir: Path, frontier: list[dict[str, Any]]) -> None:
    """写出便于人工审阅的帕累托前沿表。"""

    lines = [
        "# 参数调优帕累托前沿",
        "",
        "| setup | 产量 | 任务完整性 | 运行名 | 主要改动 | 解文件 |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in frontier:
        metrics = item["validation_metrics"]
        solution_path = (root / item["solution"]).resolve()
        full_count = metrics.get("valid_full_tasks", metrics.get("fully_scheduled_tasks", 0))
        changed = item["changed_params"] or {"baseline": True}
        changed_text = ", ".join(f"{key}={value}" for key, value in changed.items())
        lines.append(
            f"| {metrics['setup_count_positive']} | "
            f"{float(metrics['completed_weight_within_horizon']):.2f} | "
            f"{full_count}/{metrics['total_tasks']} | "
            f"`{item['name']}` | `{changed_text}` | "
            f"[{Path(item['solution']).name}]({solution_path.as_posix()}:1) |"
        )
    (output_dir / "PARETO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_files(
    root: Path,
    output_dir: Path,
    input_path: Path,
    case_slug: str,
    track: str,
    horizon: int,
    seed: int,
    grids: dict[str, list[int | float | bool]],
    cartesian: bool,
    random_trials: int,
    results: list[dict[str, Any]],
    search_mode: str = "grid",
) -> list[dict[str, Any]]:
    """写出当前已有结果的 summary.json 和 PARETO.md。

    批量调参可能持续较长时间，因此每完成一轮即刷新汇总文件。
    即使中途停止，也可以保留已完成实验点和当前帕累托前沿。
    """

    frontier = extract_frontier(results)
    summary = {
        "input": str(input_path),
        "case": case_slug,
        "track": track,
        "horizon": horizon,
        "seed": seed,
        "search_mode": search_mode,
        "grid": grids,
        "cartesian": cartesian,
        "random_trials": random_trials,
        "result_count": len(results),
        "frontier_size": len(frontier),
        "frontier": frontier,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_frontier_md(root, output_dir, frontier)
    return frontier


def relative_to_root(root: Path, path: Path) -> str:
    """将路径保存为相对项目根目录的字符串，便于跨环境查看。"""

    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def main(
    argv: Sequence[str] | None = None,
    default_track: str = "relaxed",
    forced_track: str | None = None,
) -> int:
    """批量调参主流程。

    每完成一组参数都会立即刷新 runs.jsonl、summary.json 和 PARETO.md，
    因此中途 Ctrl+C 通常只会损失正在运行的那一组。
    """

    register_pickle_compat_aliases()
    args = parse_args(argv, default_track=default_track)
    if forced_track is not None:
        args.track = forced_track
    root = args.root.resolve()
    input_path = resolve_path(root, args.input) if args.input else detect_input_json(root)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_slug = known_case_slug(input_path)
    preset: SolverPreset = CASE_PRESETS.get(case_slug, DEFAULT_PRESETS)[args.track]
    base = asdict(preset)
    base["defer_task_ids"] = sorted(base["defer_task_ids"])
    task_bonus_map = dict(preset.task_bonus_map)
    task_bonus_map.update(parse_named_float_map(args.task_bonus))
    defer_task_ids = set(preset.defer_task_ids) | parse_name_set(args.defer_task)
    force_machine_map = parse_force_machine_map(args.force_machine)

    grids = parse_grid_specs(args.grid)
    if not grids:
        grids = default_one_factor_grids(base, args.track)
    max_runs = max(1, args.max_runs)
    search_mode = "auto" if args.auto else ("cartesian" if args.cartesian else "one_factor")
    variants: list[tuple[str, dict[str, Any]]] = []
    if not args.auto:
        variants = build_variants(
            base=base,
            grids=grids,
            cartesian=args.cartesian,
            max_runs=max_runs,
            random_trials=max(0, args.random_trials),
            seed=args.seed,
        )

    instance_cache, setup_db = default_cache_paths(root, input_path, f"tune_{args.track}")
    instance = build_instance(
        root=root,
        input_path=input_path,
        cache_path=instance_cache,
        force=args.rebuild_instance,
        path_nonbatch_mult=float(base["path_nonbatch_mult"]),
        path_batch_weight=float(base["path_batch_weight"]),
        path_wait_weight=float(base["path_wait_weight"]),
        path_machine_penalties=None,
        force_path_map=None,
        current_time_override=None,
        zero_current_time=False,
        maintenance_shift=int(base["maintenance_shift"]),
    )
    if args.horizon_override is not None:
        instance.horizon = int(args.horizon_override)

    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=args.rebuild_setup)
    runs_path = output_dir / "runs.jsonl"
    results: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "input": str(input_path),
                "case": case_slug,
                "track": args.track,
                "horizon": instance.horizon,
                "search_mode": search_mode,
                "run_count": max_runs if args.auto else len(variants),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    run_index = 0

    def execute_variant(
        runs_fh: Any,
        variant_name: str,
        params: dict[str, Any],
    ) -> None:
        """执行一组参数、保存解、校验并刷新汇总文件。"""

        nonlocal run_index
        run_index += 1
        name = f"run_{run_index:03d}_{variant_name}_{variant_key(params)}"
        output_path = output_dir / f"{name}.json"
        changed_params = {
            key: value
            for key, value in params.items()
            if key in TUNABLE_FIELDS and value != base.get(key)
        }
        print(
            json.dumps(
                {
                    "event": "start_run",
                    "run": name,
                    "changed": changed_params or {"baseline": True},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        started_at = time.perf_counter()
        scheduler = RelaxedRLScheduler(
            instance,
            setup_store,
            **scheduler_kwargs(params, task_bonus_map, defer_task_ids, force_machine_map),
        )
        task_records = scheduler.solve()
        dump_solution(instance, task_records, output_path)
        solver_metrics = scheduler.metrics()
        errors, validation_metrics = run_validate(
            args.track,
            preset,
            instance,
            setup_store,
            input_path,
            output_path,
        )
        elapsed = round(time.perf_counter() - started_at, 3)
        result = {
            "name": name,
            "track": args.track,
            "search_mode": search_mode,
            "solution": relative_to_root(root, output_path),
            "elapsed_seconds": elapsed,
            "params": params,
            "changed_params": changed_params,
            "task_bonus_map": task_bonus_map,
            "defer_task_ids": sorted(defer_task_ids),
            "force_machine_count": len(force_machine_map),
            "solver_metrics": solver_metrics,
            "validation_metrics": validation_metrics,
            "error_sample": errors[:20],
        }
        results.append(result)
        runs_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        runs_fh.flush()
        write_summary_files(
            root=root,
            output_dir=output_dir,
            input_path=input_path,
            case_slug=case_slug,
            track=args.track,
            horizon=instance.horizon,
            seed=args.seed,
            grids=grids,
            cartesian=args.cartesian,
            random_trials=args.random_trials,
            results=results,
            search_mode=search_mode,
        )
        print(
            json.dumps(
                {
                    "run": name,
                    "changed": changed_params or {"baseline": True},
                    "weight": validation_metrics.get("completed_weight_within_horizon"),
                    "setup": validation_metrics.get("setup_count_positive"),
                    "errors": validation_metrics.get("error_count"),
                    "seconds": elapsed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    try:
        with runs_path.open("w", encoding="utf-8") as runs_fh:
            if args.auto:
                rng = random.Random(args.seed)
                seen: set[str] = set()
                pending: list[tuple[str, dict[str, Any]]] = []
                add_unique_variant(pending, seen, "baseline", dict(base))
                while run_index < max_runs:
                    if not pending:
                        needed = min(max(1, args.auto_round_size), max_runs - run_index)
                        pending.extend(
                            propose_auto_variants(
                                base=base,
                                results=results,
                                track=args.track,
                                seen=seen,
                                rng=rng,
                                needed=needed,
                                completed_runs=run_index,
                                max_runs=max_runs,
                                elite_size=max(1, args.auto_elite_size),
                                setup_penalty=max(0.0, args.auto_setup_penalty),
                            )
                        )
                        if not pending:
                            break
                    variant_name, params = pending.pop(0)
                    execute_variant(runs_fh, variant_name, params)
            else:
                for variant_name, params in variants:
                    execute_variant(runs_fh, variant_name, params)
    finally:
        setup_store.close()

    frontier = write_summary_files(
        root=root,
        output_dir=output_dir,
        input_path=input_path,
        case_slug=case_slug,
        track=args.track,
        horizon=instance.horizon,
        seed=args.seed,
        grids=grids,
        cartesian=args.cartesian,
        random_trials=args.random_trials,
        results=results,
        search_mode=search_mode,
    )
    print(
        json.dumps(
            {
                "frontier_size": len(frontier),
                "best_by_weight": max(
                    (
                        {
                            "name": item["name"],
                            "weight": item["validation_metrics"]["completed_weight_within_horizon"],
                            "setup": item["validation_metrics"]["setup_count_positive"],
                        }
                        for item in frontier
                    ),
                    key=lambda item: (item["weight"], -item["setup"]),
                    default=None,
                ),
                "pareto_md": str(output_dir / "PARETO.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

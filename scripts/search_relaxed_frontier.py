#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from rl_relaxed_solver import (
    RelaxedRLScheduler,
    SetupRowStore,
    build_instance,
    detect_input_json,
    dump_solution,
    validate_solution,
)


BASE_PARAMS: dict[str, float | int] = {
    "lookahead": 70,
    "start_guard": 330,
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
}

BASE_PATH_PARAMS: dict[str, Any] = {
    "path_nonbatch_mult": 3.0,
    "path_batch_weight": 1.0,
    "path_wait_weight": 1.0,
    "path_machine_penalties": {},
}

PATH_PENALTY_MACHINES = ("汽车板连退线", "汽车板2#连退线")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a Pareto frontier around the relaxed solver")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--horizon-override", type=int, default=24480)
    parser.add_argument("--instance-cache", type=Path, default=Path("cache/actual_scale.pkl"))
    parser.add_argument("--setup-db", type=Path, default=Path("cache/actual_scale.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/auto_frontier_24480"))
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefix", type=str, default="auto")
    parser.add_argument("--anchor-file", type=Path, default=None, help="Previous search_results.json to use as local-search anchors")
    parser.add_argument("--anchor-count", type=int, default=4, help="How many anchor points to sample from")
    parser.add_argument("--explore-scale", type=float, default=1.0, help="Scale factor for parameter perturbation around an anchor")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip re-running the fixed baseline point")
    parser.add_argument(
        "--force-path",
        type=str,
        default="",
        help="Comma-separated task_id=path_id overrides applied to every search run",
    )
    return parser.parse_args()


def as_rel_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sample_params(
    rng: random.Random,
    anchor: dict[str, float | int] | None = None,
    explore_scale: float = 1.0,
) -> dict[str, float | int]:
    base = dict(BASE_PARAMS if anchor is None else anchor)
    scale = max(explore_scale, 0.1)
    params = dict(base)
    params["lookahead"] = clamp_int(int(round(float(base["lookahead"]))) + rng.randint(int(round(-14 * scale)), int(round(14 * scale))), 40, 110)
    params["start_guard"] = clamp_int(int(round(float(base["start_guard"]))) + rng.randint(int(round(-180 * scale)), int(round(180 * scale))), 120, 720)
    params["score_weight"] = clamp_float(float(base["score_weight"]) + rng.uniform(-20.0, 20.0) * scale, 260.0, 420.0)
    params["score_density"] = clamp_float(float(base["score_density"]) + rng.uniform(-2500.0, 2500.0) * scale, 18000.0, 28000.0)
    params["score_started"] = clamp_float(float(base["score_started"]) + rng.uniform(-40.0, 60.0) * scale, 40.0, 260.0)
    params["score_family"] = clamp_float(float(base["score_family"]) + rng.uniform(-140.0, 160.0) * scale, 120.0, 700.0)
    params["score_zero_setup"] = clamp_float(float(base["score_zero_setup"]) + rng.uniform(-40.0, 200.0) * scale, 0.0, 500.0)
    params["score_setup_fixed"] = clamp_float(float(base["score_setup_fixed"]) + rng.uniform(-80.0, 90.0) * scale, 220.0, 620.0)
    params["score_setup_per"] = clamp_float(float(base["score_setup_per"]) + rng.uniform(-0.8, 0.8) * scale, 2.5, 6.5)
    params["score_est_final_per"] = clamp_float(float(base["score_est_final_per"]) + rng.uniform(-0.008, 0.012) * scale, 0.0, 0.05)
    params["phase2_started"] = clamp_float(float(base["phase2_started"]) + rng.uniform(-350.0, 350.0) * scale, 1200.0, 3200.0)
    params["phase2_density"] = clamp_float(float(base["phase2_density"]) + rng.uniform(-1000.0, 1000.0) * scale, 4200.0, 8600.0)
    params["phase2_family"] = clamp_float(float(base["phase2_family"]) + rng.uniform(-160.0, 160.0) * scale, 180.0, 900.0)
    params["phase2_zero_setup"] = clamp_float(float(base["phase2_zero_setup"]) + rng.uniform(-80.0, 240.0) * scale, 0.0, 1200.0)
    params["phase2_setup_fixed"] = clamp_float(float(base["phase2_setup_fixed"]) + rng.uniform(-80.0, 80.0) * scale, 260.0, 760.0)
    params["phase2_setup_per"] = clamp_float(float(base["phase2_setup_per"]) + rng.uniform(-0.8, 0.8) * scale, 2.5, 6.5)
    params["phase2_finish_per"] = clamp_float(float(base["phase2_finish_per"]) + rng.uniform(-0.01, 0.02) * scale, 0.0, 0.06)
    return params


def normalize_path_params(path_params: dict[str, Any]) -> dict[str, Any]:
    penalties = {
        machine: float(value)
        for machine, value in path_params.get("path_machine_penalties", {}).items()
        if float(value) > 1e-9
    }
    return {
        "path_nonbatch_mult": float(path_params.get("path_nonbatch_mult", BASE_PATH_PARAMS["path_nonbatch_mult"])),
        "path_batch_weight": float(path_params.get("path_batch_weight", BASE_PATH_PARAMS["path_batch_weight"])),
        "path_wait_weight": float(path_params.get("path_wait_weight", BASE_PATH_PARAMS["path_wait_weight"])),
        "path_machine_penalties": dict(sorted(penalties.items())),
    }


def parse_force_path(force_path_arg: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in (part.strip() for part in force_path_arg.split(",")):
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid force-path override: {chunk}")
        task_id, path_id = chunk.split("=", 1)
        task_id = task_id.strip()
        path_id = path_id.strip()
        if not task_id or not path_id:
            raise ValueError(f"Invalid force-path override: {chunk}")
        mapping[task_id] = path_id
    return mapping


def sample_path_params(
    rng: random.Random,
    anchor: dict[str, Any] | None = None,
    explore_scale: float = 1.0,
) -> dict[str, Any]:
    base = normalize_path_params(BASE_PATH_PARAMS if anchor is None else anchor)
    scale = max(explore_scale, 0.1)
    result = {
        "path_nonbatch_mult": clamp_float(base["path_nonbatch_mult"] + rng.uniform(-0.7, 0.8) * scale, 1.5, 5.5),
        "path_batch_weight": clamp_float(base["path_batch_weight"] + rng.uniform(-0.35, 0.35) * scale, 0.4, 2.2),
        "path_wait_weight": clamp_float(base["path_wait_weight"] + rng.uniform(-0.35, 0.45) * scale, 0.3, 2.5),
        "path_machine_penalties": {},
    }
    anchor_penalties = base["path_machine_penalties"]
    penalties: dict[str, float] = {}
    for machine in PATH_PENALTY_MACHINES:
        anchor_value = float(anchor_penalties.get(machine, 0.0))
        candidate_value = clamp_float(anchor_value + rng.uniform(-90.0, 120.0) * scale, 0.0, 420.0)
        if candidate_value >= 20.0:
            penalties[machine] = round(candidate_value, 3)
    result["path_machine_penalties"] = penalties
    return normalize_path_params(result)


def load_anchor_params(anchor_file: Path, anchor_count: int) -> list[dict[str, float | int]]:
    with anchor_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    anchors: list[dict[str, float | int]] = [dict(BASE_PARAMS)]
    frontier_items = payload.get("frontier", [])
    sorted_items = sorted(
        frontier_items,
        key=lambda item: (
            -float(item["metrics"]["completed_weight_within_horizon"]),
            float(item["metrics"]["setup_count_positive"]),
            item["name"],
        ),
    )
    for item in sorted_items[: max(anchor_count, 0)]:
        params = item.get("params")
        if params:
            anchors.append(params)
    unique: list[dict[str, float | int]] = []
    seen: set[str] = set()
    for item in anchors:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def load_anchor_path_params(anchor_file: Path, anchor_count: int) -> list[dict[str, Any]]:
    with anchor_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    anchors: list[dict[str, Any]] = [dict(BASE_PATH_PARAMS)]
    frontier_items = payload.get("frontier", [])
    sorted_items = sorted(
        frontier_items,
        key=lambda item: (
            -float(item["metrics"]["completed_weight_within_horizon"]),
            float(item["metrics"]["setup_count_positive"]),
            item["name"],
        ),
    )
    for item in sorted_items[: max(anchor_count, 0)]:
        params = item.get("path_params")
        if params:
            anchors.append(normalize_path_params(params))
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in anchors:
        key = json.dumps(normalize_path_params(item), sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalize_path_params(item))
    return unique


def is_valid_result(result: dict[str, Any]) -> bool:
    metrics = result["metrics"]
    validation = result["validation"]
    return (
        metrics["fully_scheduled_tasks"] == metrics["total_tasks"]
        and validation["error_count"] == 0
    )


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm = left["metrics"]
    rm = right["metrics"]
    no_worse = (
        lm["setup_count_positive"] <= rm["setup_count_positive"]
        and lm["completed_weight_within_horizon"] >= rm["completed_weight_within_horizon"]
    )
    strictly_better = (
        lm["setup_count_positive"] < rm["setup_count_positive"]
        or lm["completed_weight_within_horizon"] > rm["completed_weight_within_horizon"]
    )
    return no_worse and strictly_better


def extract_frontier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_results = [item for item in results if is_valid_result(item)]
    frontier: list[dict[str, Any]] = []
    for candidate in valid_results:
        if any(dominates(other, candidate) for other in valid_results if other is not candidate):
            continue
        frontier.append(candidate)
    frontier.sort(
        key=lambda item: (
            item["metrics"]["setup_count_positive"],
            -item["metrics"]["completed_weight_within_horizon"],
            item["name"],
        )
    )
    return frontier


def format_solution_name(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return (
        f"{result['name']}_"
        f"{metrics['completed_weight_within_horizon']:.2f}_"
        f"{metrics['setup_count_positive']}.json"
    ).replace(":", "_")


def strategy_cache_path(instance_cache: Path, path_params: dict[str, Any], force_path_map: dict[str, str]) -> Path:
    payload = json.dumps(
        {
            "path_params": normalize_path_params(path_params),
            "force_path_map": dict(sorted(force_path_map.items())),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
    return instance_cache.with_name(f"{instance_cache.stem}_{digest}{instance_cache.suffix}")


def extract_path_ids(instance: Any) -> dict[str, str]:
    return {task_id: task.path_id for task_id, task in instance.tasks.items()}


def summarize_path_changes(path_ids: dict[str, str], baseline_path_ids: dict[str, str]) -> dict[str, Any]:
    changed = sorted(task_id for task_id, path_id in path_ids.items() if baseline_path_ids.get(task_id) != path_id)
    return {
        "path_change_count": len(changed),
        "path_change_sample": changed[:20],
    }


def run_one(
    root: Path,
    output_dir: Path,
    name: str,
    input_path: Path,
    instance_cache: Path,
    setup_store: SetupRowStore,
    params: dict[str, float | int],
    path_params: dict[str, Any],
    force_path_map: dict[str, str],
    horizon: int,
    baseline_path_ids: dict[str, str],
) -> dict[str, Any]:
    cache_path = strategy_cache_path(instance_cache, path_params, force_path_map)
    instance = build_instance(
        root,
        input_path,
        cache_path,
        force=False,
        path_nonbatch_mult=float(path_params["path_nonbatch_mult"]),
        path_batch_weight=float(path_params["path_batch_weight"]),
        path_wait_weight=float(path_params["path_wait_weight"]),
        path_machine_penalties=dict(path_params["path_machine_penalties"]),
        force_path_map=force_path_map,
    )
    instance.horizon = int(horizon)
    scheduler = RelaxedRLScheduler(instance, setup_store, **params)
    task_records = scheduler.solve()
    metrics = scheduler.metrics()
    solution_path = output_dir / format_solution_name({"name": name, "metrics": metrics})
    dump_solution(instance, task_records, solution_path)
    errors, validation = validate_solution(instance, setup_store, solution_path)
    result = {
        "name": name,
        "params": params,
        "path_params": normalize_path_params(path_params),
        "force_path_map": dict(sorted(force_path_map.items())),
        **summarize_path_changes(extract_path_ids(instance), baseline_path_ids),
        "metrics": metrics,
        "validation": validation,
        "solution": as_rel_path(root, solution_path),
    }
    if errors:
        result["errors_preview"] = errors[:10]
    return result


def write_frontier_md(root: Path, output_dir: Path, frontier: list[dict[str, Any]]) -> None:
    lines = [
        f"Auto frontier search for horizon `24480`.",
        "",
        "| setup_count_positive | completed_weight_within_horizon | path_change_count | run | solution |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in frontier:
        metrics = item["metrics"]
        solution_path = (root / item["solution"]).resolve()
        lines.append(
            f"| {metrics['setup_count_positive']} | {metrics['completed_weight_within_horizon']:.2f} | {item['path_change_count']} | "
            f"`{item['name']}` | [{Path(item['solution']).name}]({solution_path.as_posix()}:1) |"
        )
    (output_dir / "FRONTIER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    root = args.root.resolve()
    input_path = args.input.resolve() if args.input else detect_input_json(root)
    instance_cache = (root / args.instance_cache).resolve() if not args.instance_cache.is_absolute() else args.instance_cache
    setup_db = (root / args.setup_db).resolve() if not args.setup_db.is_absolute() else args.setup_db
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    anchor_file = None
    if args.anchor_file is not None:
        anchor_file = (root / args.anchor_file).resolve() if not args.anchor_file.is_absolute() else args.anchor_file
    output_dir.mkdir(parents=True, exist_ok=True)
    force_path_map = parse_force_path(args.force_path)

    instance = build_instance(root, input_path, instance_cache, force=False)
    if args.horizon_override is not None:
        instance.horizon = int(args.horizon_override)
    baseline_path_ids = extract_path_ids(instance)
    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=False)

    try:
        results: list[dict[str, Any]] = []
        anchors = [dict(BASE_PARAMS)]
        path_anchors = [dict(BASE_PATH_PARAMS)]
        if anchor_file is not None and anchor_file.exists():
            anchors = load_anchor_params(anchor_file, args.anchor_count)
            path_anchors = load_anchor_path_params(anchor_file, args.anchor_count)
        if not args.skip_baseline:
            baseline = run_one(
                root,
                output_dir,
                f"{args.prefix}_baseline",
                input_path,
                instance_cache,
                setup_store,
                dict(BASE_PARAMS),
                dict(BASE_PATH_PARAMS),
                force_path_map,
                instance.horizon,
                baseline_path_ids,
            )
            print(json.dumps({"baseline": baseline["metrics"], "validation": baseline["validation"]}, ensure_ascii=False, indent=2), flush=True)
            results.append(baseline)

        for idx in range(1, args.trials + 1):
            anchor = anchors[(idx - 1) % len(anchors)] if anchors else None
            path_anchor = path_anchors[(idx - 1) % len(path_anchors)] if path_anchors else None
            params = sample_params(rng, anchor=anchor, explore_scale=args.explore_scale)
            path_params = sample_path_params(rng, anchor=path_anchor, explore_scale=args.explore_scale)
            name = f"{args.prefix}_{idx:02d}"
            result = run_one(
                root,
                output_dir,
                name,
                input_path,
                instance_cache,
                setup_store,
                params,
                path_params,
                force_path_map,
                instance.horizon,
                baseline_path_ids,
            )
            print(
                json.dumps(
                    {
                        "run": name,
                        "anchor": "baseline" if anchor is None or anchor == BASE_PARAMS else "frontier",
                        "path_anchor": "baseline" if path_anchor is None or normalize_path_params(path_anchor) == normalize_path_params(BASE_PATH_PARAMS) else "frontier",
                        "path_params": result["path_params"],
                        "path_change_count": result["path_change_count"],
                        "metrics": result["metrics"],
                        "validation": result["validation"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            results.append(result)

        frontier = extract_frontier(results)
        summary = {
            "input": str(input_path),
            "horizon": instance.horizon,
            "seed": args.seed,
            "trials": args.trials,
            "explore_scale": args.explore_scale,
            "anchor_file": str(anchor_file) if anchor_file is not None else None,
            "force_path_map": dict(sorted(force_path_map.items())),
            "frontier_size": len(frontier),
            "frontier": [
                {
                    "name": item["name"],
                    "solution": item["solution"],
                    "metrics": item["metrics"],
                    "validation": item["validation"],
                    "params": item["params"],
                    "path_params": item["path_params"],
                    "force_path_map": item["force_path_map"],
                    "path_change_count": item["path_change_count"],
                    "path_change_sample": item["path_change_sample"],
                }
                for item in frontier
            ],
            "results": results,
        }
        (output_dir / "search_results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_frontier_md(root, output_dir, frontier)
        print(json.dumps({"frontier_size": len(frontier), "frontier_best": frontier[-1]["metrics"] if frontier else None}, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        setup_store.close()


if __name__ == "__main__":
    raise SystemExit(main())

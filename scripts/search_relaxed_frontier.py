#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    "start_guard": 360,
    "score_weight": 340.0,
    "score_density": 23500.0,
    "score_started": 140.0,
    "score_family": 400.0,
    "score_progress": 0.0,
    "score_zero_setup": 0.0,
    "score_setup_fixed": 390.0,
    "score_setup_per": 4.0,
    "phase2_started": 2200.0,
    "phase2_density": 6900.0,
    "phase2_family": 560.0,
    "phase2_progress": 0.0,
    "phase2_zero_setup": 0.0,
    "phase2_setup_fixed": 500.0,
    "phase2_setup_per": 4.0,
}


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
    return parser.parse_args()


def as_rel_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def sample_params(rng: random.Random) -> dict[str, float | int]:
    params = dict(BASE_PARAMS)
    params["lookahead"] = max(40, min(110, int(BASE_PARAMS["lookahead"]) + rng.randint(-14, 14)))
    params["start_guard"] = max(120, min(720, int(BASE_PARAMS["start_guard"]) + rng.randint(-180, 180)))
    params["score_weight"] = float(BASE_PARAMS["score_weight"]) + rng.uniform(-20.0, 20.0)
    params["score_density"] = float(BASE_PARAMS["score_density"]) + rng.uniform(-2500.0, 2500.0)
    params["score_started"] = float(BASE_PARAMS["score_started"]) + rng.uniform(-40.0, 60.0)
    params["score_family"] = float(BASE_PARAMS["score_family"]) + rng.uniform(-140.0, 160.0)
    params["score_zero_setup"] = max(0.0, float(BASE_PARAMS["score_zero_setup"]) + rng.uniform(-40.0, 200.0))
    params["score_setup_fixed"] = float(BASE_PARAMS["score_setup_fixed"]) + rng.uniform(-80.0, 90.0)
    params["score_setup_per"] = max(2.5, float(BASE_PARAMS["score_setup_per"]) + rng.uniform(-0.8, 0.8))
    params["phase2_started"] = float(BASE_PARAMS["phase2_started"]) + rng.uniform(-350.0, 350.0)
    params["phase2_density"] = float(BASE_PARAMS["phase2_density"]) + rng.uniform(-1000.0, 1000.0)
    params["phase2_family"] = float(BASE_PARAMS["phase2_family"]) + rng.uniform(-160.0, 160.0)
    params["phase2_zero_setup"] = max(0.0, float(BASE_PARAMS["phase2_zero_setup"]) + rng.uniform(-80.0, 240.0))
    params["phase2_setup_fixed"] = float(BASE_PARAMS["phase2_setup_fixed"]) + rng.uniform(-80.0, 80.0)
    params["phase2_setup_per"] = max(2.5, float(BASE_PARAMS["phase2_setup_per"]) + rng.uniform(-0.8, 0.8))
    return params


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


def run_one(
    root: Path,
    output_dir: Path,
    name: str,
    instance,
    setup_store: SetupRowStore,
    params: dict[str, float | int],
) -> dict[str, Any]:
    scheduler = RelaxedRLScheduler(instance, setup_store, **params)
    task_records = scheduler.solve()
    metrics = scheduler.metrics()
    solution_path = output_dir / format_solution_name({"name": name, "metrics": metrics})
    dump_solution(instance, task_records, solution_path)
    errors, validation = validate_solution(instance, setup_store, solution_path)
    result = {
        "name": name,
        "params": params,
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
        "| setup_count_positive | completed_weight_within_horizon | run | solution |",
        "| --- | ---: | --- | --- |",
    ]
    for item in frontier:
        metrics = item["metrics"]
        solution_path = (root / item["solution"]).resolve()
        lines.append(
            f"| {metrics['setup_count_positive']} | {metrics['completed_weight_within_horizon']:.2f} | "
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
    output_dir.mkdir(parents=True, exist_ok=True)

    instance = build_instance(root, input_path, instance_cache, force=False)
    if args.horizon_override is not None:
        instance.horizon = int(args.horizon_override)
    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=False)

    try:
        results: list[dict[str, Any]] = []
        baseline = run_one(root, output_dir, f"{args.prefix}_baseline", instance, setup_store, dict(BASE_PARAMS))
        print(json.dumps({"baseline": baseline["metrics"], "validation": baseline["validation"]}, ensure_ascii=False, indent=2), flush=True)
        results.append(baseline)

        for idx in range(1, args.trials + 1):
            params = sample_params(rng)
            name = f"{args.prefix}_{idx:02d}"
            result = run_one(root, output_dir, name, instance, setup_store, params)
            print(json.dumps({"run": name, "metrics": result["metrics"], "validation": result["validation"]}, ensure_ascii=False, indent=2), flush=True)
            results.append(result)

        frontier = extract_frontier(results)
        summary = {
            "input": str(input_path),
            "horizon": instance.horizon,
            "seed": args.seed,
            "trials": args.trials,
            "frontier_size": len(frontier),
            "frontier": [
                {
                    "name": item["name"],
                    "solution": item["solution"],
                    "metrics": item["metrics"],
                    "validation": item["validation"],
                    "params": item["params"],
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from rl_relaxed_solver import SetupRowStore, build_instance, detect_input_json
from search_relaxed_frontier import (
    BASE_PARAMS,
    BASE_PATH_PARAMS,
    as_rel_path,
    extract_frontier,
    extract_path_ids,
    load_anchor_params,
    normalize_path_params,
    run_one,
    sample_params,
)


DEFAULT_BUNDLES: list[dict[str, Any]] = [
    {"name": "baseline", "force_path": {}},
    {"name": "yt0363", "force_path": {"YT0363": "0"}},
    {"name": "kb0043", "force_path": {"KB0043": "1"}},
    {"name": "kb0032_kb0043", "force_path": {"KB0032": "1", "KB0043": "1"}},
    {"name": "kb0039_kb0043", "force_path": {"KB0039": "1", "KB0043": "1"}},
    {"name": "yt0363_kb0043", "force_path": {"YT0363": "0", "KB0043": "1"}},
    {
        "name": "yt0363_bundle5",
        "force_path": {"YT0363": "0", "KB0032": "1", "KB0037": "1", "KB0038": "1", "KB0043": "1"},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search multiple fixed force-path bundles")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--horizon-override", type=int, default=24480)
    parser.add_argument("--instance-cache", type=Path, default=Path("cache/actual_scale.pkl"))
    parser.add_argument("--setup-db", type=Path, default=Path("cache/actual_scale.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/auto_force_bundle_search"))
    parser.add_argument("--trials-per-bundle", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--anchor-file", type=Path, default=None)
    parser.add_argument("--anchor-count", type=int, default=3)
    parser.add_argument("--explore-scale", type=float, default=0.2)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--bundles-file", type=Path, default=None, help="JSON file with [{'name':..., 'force_path': {...}}]")
    return parser.parse_args()


def load_bundles(bundles_file: Path | None) -> list[dict[str, Any]]:
    if bundles_file is None:
        return [dict(item) for item in DEFAULT_BUNDLES]
    with bundles_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    bundles: list[dict[str, Any]] = []
    for item in payload:
        name = str(item["name"])
        force_path = {str(task_id): str(path_id) for task_id, path_id in item.get("force_path", {}).items()}
        bundles.append({"name": name, "force_path": force_path})
    return bundles


def write_frontier_md(root: Path, output_dir: Path, frontier: list[dict[str, Any]]) -> None:
    lines = [
        "Auto search across fixed force-path bundles.",
        "",
        "| setup_count_positive | completed_weight_within_horizon | bundle | run | solution |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for item in frontier:
        metrics = item["metrics"]
        solution_path = (root / item["solution"]).resolve()
        lines.append(
            f"| {metrics['setup_count_positive']} | {metrics['completed_weight_within_horizon']:.2f} | "
            f"`{item['bundle_name']}` | `{item['name']}` | [{Path(item['solution']).name}]({solution_path.as_posix()}:1) |"
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
    bundles_file = None
    if args.bundles_file is not None:
        bundles_file = (root / args.bundles_file).resolve() if not args.bundles_file.is_absolute() else args.bundles_file
    anchor_file = None
    if args.anchor_file is not None:
        anchor_file = (root / args.anchor_file).resolve() if not args.anchor_file.is_absolute() else args.anchor_file
    output_dir.mkdir(parents=True, exist_ok=True)

    instance = build_instance(root, input_path, instance_cache, force=False)
    if args.horizon_override is not None:
        instance.horizon = int(args.horizon_override)
    baseline_path_ids = extract_path_ids(instance)
    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=False)

    anchors = [dict(BASE_PARAMS)]
    if anchor_file is not None and anchor_file.exists():
        anchors = load_anchor_params(anchor_file, args.anchor_count)
    bundles = load_bundles(bundles_file)

    all_results: list[dict[str, Any]] = []
    try:
        for bundle_idx, bundle in enumerate(bundles):
            bundle_name = bundle["name"]
            force_path = bundle["force_path"]
            bundle_dir = output_dir / bundle_name
            bundle_dir.mkdir(parents=True, exist_ok=True)

            if not args.skip_baseline:
                baseline = run_one(
                    root,
                    bundle_dir,
                    f"{bundle_name}_baseline",
                    input_path,
                    instance_cache,
                    setup_store,
                    dict(BASE_PARAMS),
                    dict(BASE_PATH_PARAMS),
                    force_path,
                    instance.horizon,
                    baseline_path_ids,
                )
                baseline["bundle_name"] = bundle_name
                all_results.append(baseline)
                print(
                    json.dumps(
                        {"bundle": bundle_name, "baseline": baseline["metrics"], "validation": baseline["validation"]},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    flush=True,
                )

            local_rng = random.Random(args.seed + bundle_idx * 1009)
            for trial_idx in range(1, args.trials_per_bundle + 1):
                anchor = anchors[(trial_idx - 1) % len(anchors)] if anchors else None
                params = sample_params(local_rng, anchor=anchor, explore_scale=args.explore_scale)
                result = run_one(
                    root,
                    bundle_dir,
                    f"{bundle_name}_{trial_idx:02d}",
                    input_path,
                    instance_cache,
                    setup_store,
                    params,
                    normalize_path_params(BASE_PATH_PARAMS),
                    force_path,
                    instance.horizon,
                    baseline_path_ids,
                )
                result["bundle_name"] = bundle_name
                all_results.append(result)
                print(
                    json.dumps(
                        {
                            "bundle": bundle_name,
                            "run": result["name"],
                            "force_path_count": len(force_path),
                            "metrics": result["metrics"],
                            "validation": result["validation"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    flush=True,
                )

        frontier = extract_frontier(all_results)
        summary = {
            "input": str(input_path),
            "horizon": instance.horizon,
            "seed": args.seed,
            "trials_per_bundle": args.trials_per_bundle,
            "explore_scale": args.explore_scale,
            "anchor_file": str(anchor_file) if anchor_file is not None else None,
            "bundles": bundles,
            "frontier_size": len(frontier),
            "frontier": [
                {
                    "bundle_name": item["bundle_name"],
                    "name": item["name"],
                    "solution": item["solution"],
                    "metrics": item["metrics"],
                    "validation": item["validation"],
                    "params": item["params"],
                    "path_params": item["path_params"],
                    "force_path_map": item["force_path_map"],
                }
                for item in frontier
            ],
            "results": all_results,
        }
        (output_dir / "search_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_frontier_md(root, output_dir, frontier)
        print(
            json.dumps(
                {
                    "frontier_size": len(frontier),
                    "frontier_best": frontier[-1]["metrics"] if frontier else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        setup_store.close()


if __name__ == "__main__":
    raise SystemExit(main())

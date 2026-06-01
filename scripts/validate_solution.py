#!/usr/bin/env python3
"""relaxed 口径解文件校验入口。

relaxed 表示组批机器按无限产能处理；普通机器、任务内部顺序、维修、
释放时间、q-time、跨厂转运和 setup 仍严格校验。有限组批解应使用
validate_batch_solution.py。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rl_relaxed_solver import (
    SetupRowStore,
    build_instance,
    detect_input_json,
    infer_force_path_map_from_solution,
    parse_named_float_map,
    parse_named_str_map,
    validate_solution,
)


def resolve_path(root: Path, path: Path) -> Path:
    """把命令行相对路径统一解析到项目根目录下。"""
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_args() -> argparse.Namespace:
    """解析 relaxed 校验器需要的命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Huawei FJSP solution JSON under the relaxed infinite-batch "
            "capacity rule used by this project."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root. Relative paths are resolved against this directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input instance JSON. Defaults to data/data1/*.json under --root.",
    )
    parser.add_argument(
        "--solution",
        "--output",
        dest="solution",
        type=Path,
        default=Path("outputs/rl_relaxed_solution.json"),
        help="Solution JSON to validate.",
    )
    parser.add_argument(
        "--horizon",
        "--horizon-override",
        dest="horizon_override",
        type=int,
        default=None,
        help="Override config.max_output_horizon for metric calculation.",
    )
    parser.add_argument(
        "--instance-cache",
        type=Path,
        default=Path("cache/validation_instance.pkl"),
        help="Pickle cache for parsed instance data.",
    )
    parser.add_argument(
        "--setup-db",
        type=Path,
        default=Path("cache/setup_rows.sqlite"),
        help="SQLite cache for sparse setup matrix rows.",
    )
    parser.add_argument("--path-nonbatch-mult", type=float, default=3.0)
    parser.add_argument("--path-batch-weight", type=float, default=1.0)
    parser.add_argument("--path-wait-weight", type=float, default=1.0)
    parser.add_argument(
        "--path-machine-penalty",
        type=str,
        default="",
        help="Comma-separated machine=penalty pairs used if paths must be selected.",
    )
    parser.add_argument(
        "--force-path",
        type=str,
        default="",
        help=(
            "Comma-separated task_id=path_id overrides. If omitted, the script "
            "infers task paths from the solution file."
        ),
    )
    parser.add_argument(
        "--no-infer-force-path",
        action="store_true",
        help="Do not infer task path choices from the solution file.",
    )
    parser.add_argument("--rebuild-instance", action="store_true")
    parser.add_argument("--rebuild-setup", action="store_true")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=-1,
        help="Maximum validation errors to print. Use -1 to print all errors.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path containing metrics and sampled errors.",
    )
    return parser.parse_args()


def main() -> int:
    """命令行主流程。

    执行顺序：
    1. 解析输入算例、解文件、缓存路径和可选路径参数。
    2. 如未显式传入 --force-path，则从解文件自动推断每个任务的 path_id。
    3. 构建与解文件一致的 InstanceData，并可按 --horizon 覆盖统计截止时间。
    4. 加载 setup 行缓存，调用 relaxed 核心校验函数。
    5. 打印指标、错误明细，并在需要时写出 JSON 报告。

    返回码约定：0 表示校验通过，1 表示发现约束错误，2 表示输入文件缺失。
    """
    args = parse_args()
    root = args.root.resolve()
    input_path = args.input.resolve() if args.input else detect_input_json(root)
    solution_path = resolve_path(root, args.solution)
    instance_cache = resolve_path(root, args.instance_cache)
    setup_db = resolve_path(root, args.setup_db)

    # 输入和解文件缺失时直接返回 2，便于脚本化批量校验时区分运行错误。
    if not input_path.exists():
        print(f"[error] input JSON not found: {input_path}", file=sys.stderr)
        return 2
    if not solution_path.exists():
        print(f"[error] solution JSON not found: {solution_path}", file=sys.stderr)
        return 2

    path_machine_penalties = parse_named_float_map(args.path_machine_penalty)
    force_path_map = parse_named_str_map(args.force_path)
    inferred_force_paths = False
    if not force_path_map and not args.no_infer_force_path:
        # 解文件中每道工序都带 path_id，因此通常可以自动恢复任务选路，
        # 不需要用户手动重复很长的 --force-path 参数。
        force_path_map = infer_force_path_map_from_solution(solution_path)
        inferred_force_paths = bool(force_path_map)

    print(f"[validate] input: {input_path}", flush=True)
    print(f"[validate] solution: {solution_path}", flush=True)
    if inferred_force_paths:
        print(
            f"[validate] inferred force paths from solution: {len(force_path_map)}",
            flush=True,
        )

    instance = build_instance(
        root,
        input_path,
        instance_cache,
        force=args.rebuild_instance,
        path_nonbatch_mult=args.path_nonbatch_mult,
        path_batch_weight=args.path_batch_weight,
        path_wait_weight=args.path_wait_weight,
        path_machine_penalties=path_machine_penalties,
        force_path_map=force_path_map,
    )
    if args.horizon_override is not None:
        # 只覆盖指标统计/校验使用的截止时间，不改写输入 JSON。
        instance.horizon = int(args.horizon_override)
        print(f"[validate] horizon override: {instance.horizon}", flush=True)

    # 切换矩阵很大，校验时按行存进 SQLite 缓存，避免每次整体加载。
    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=args.rebuild_setup)
    try:
        errors, metrics = validate_solution(instance, setup_store, solution_path)
    finally:
        setup_store.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    shown_errors = errors if args.max_errors < 0 else errors[: args.max_errors]
    if errors:
        print(f"[validate] errors: {len(errors)}", flush=True)
        if args.max_errors >= 0 and len(errors) > len(shown_errors):
            print(
                f"[validate] showing first {len(shown_errors)} errors; "
                "rerun with --max-errors -1 to print all",
                flush=True,
            )
        for idx, item in enumerate(shown_errors, start=1):
            print(f"[{idx:04d}] {item}", flush=True)
    else:
        print("[validate] no errors", flush=True)

    if args.report is not None:
        # 报告文件用于保存批量实验的校验指标和错误样例，便于后续对比。
        report_path = resolve_path(root, args.report)
        report: dict[str, Any] = {
            "input": str(input_path),
            "solution": str(solution_path),
            "horizon": instance.horizon,
            "inferred_force_path_count": len(force_path_map) if inferred_force_paths else 0,
            "metrics": metrics,
            "errors": shown_errors,
            "error_count": len(errors),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"[validate] report: {report_path}", flush=True)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

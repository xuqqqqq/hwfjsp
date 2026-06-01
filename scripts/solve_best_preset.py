#!/usr/bin/env python3
"""预设求解入口。

本文件将常用参数固化为 preset，避免在常规运行中重复输入长命令。
具体派工由 rl_relaxed_solver.RelaxedRLScheduler 完成；本文件只负责：
选择算例预设、构建缓存、生成解文件，并调用对应校验器。
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rl_relaxed_solver import (
    RelaxedRLScheduler,
    SetupRowStore,
    build_instance,
    detect_input_json,
    dump_solution,
    infer_force_path_map_from_solution,
    register_pickle_compat_aliases,
    validate_solution,
)
from validate_batch_solution import validate_batch_solution


@dataclass(frozen=True)
class SolverPreset:
    """一组可复现实验参数。

    字段命名和 RelaxedRLScheduler 构造参数保持一致，便于从 preset
    可直接传入求解器。
    """

    # 输入 JSON 是时间轴的唯一来源。求解器把 time.current_time 当作当前时刻，
    # 并使用 horizon、维修窗口、释放时间、q-time 和转运时间。
    maintenance_shift: int = 0
    path_nonbatch_mult: float = 3.0
    path_batch_weight: float = 1.0
    path_wait_weight: float = 1.0
    lookahead: int = 85
    start_guard: int = 120
    score_weight: float = 340.0
    score_density: float = 23500.0
    score_started: float = 140.0
    score_family: float = 400.0
    score_progress: float = 0.0
    score_zero_setup: float = 380.0
    score_setup_fixed: float = 390.0
    score_setup_per: float = 4.0
    score_est_final_per: float = 0.01
    task_bonus_map: dict[str, float] = field(default_factory=dict)
    defer_task_ids: set[str] = field(default_factory=set)
    phase2_started: float = 2200.0
    phase2_density: float = 6900.0
    phase2_family: float = 560.0
    phase2_progress: float = 0.0
    phase2_zero_setup: float = 900.0
    phase2_setup_fixed: float = 500.0
    phase2_setup_per: float = 4.0
    phase2_finish_per: float = 0.01
    phase2_allow_unstarted: bool = True
    finite_batch_capacity: bool = False
    batch_group_wait: int = 0
    batch_group_mixed_time: bool = False
    batch_group_any_time: bool = False


DEFAULT_PRESETS: dict[str, SolverPreset] = {
    # 未知/新算例使用通用参数。此处不放置任务 id 级别的 bonus，
    # 因为任务 id 是算例相关的。
    "relaxed": SolverPreset(),
    "finite": SolverPreset(
        finite_batch_capacity=True,
        batch_group_wait=320,
        batch_group_mixed_time=True,
    ),
}

CASE_PRESETS: dict[str, dict[str, SolverPreset]] = {
    "test_case1": {
        # 测试算例1在当前 JSON 时间口径下的 relaxed 最优固化参数。
        "relaxed": SolverPreset(
            score_zero_setup=0.0,
            phase2_zero_setup=0.0,
            task_bonus_map={"KB0039": 100.0},
        ),
        # 测试算例1的有限组批最优固化参数，最终校验必须走 validate_batch_solution.py。
        "finite": SolverPreset(
            score_zero_setup=0.0,
            phase2_zero_setup=0.0,
            finite_batch_capacity=True,
            batch_group_wait=320,
            batch_group_mixed_time=True,
        ),
    },
    "test_case2": {
        # 测试算例2在当前 JSON 时间口径下的 relaxed 固化参数。
        "relaxed": SolverPreset(
            lookahead=75,
            task_bonus_map={
                "NP8889": 40.0,
                "NP8887": 40.0,
                "NP8886": 40.0,
            },
        ),
        # 测试算例2更适合同 family 的 any-time 组批策略。
        "finite": SolverPreset(
            finite_batch_capacity=True,
            batch_group_wait=300,
            batch_group_any_time=True,
        ),
    },
}


def parse_args() -> argparse.Namespace:
    """解析预设求解入口的命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Run the built-in Huawei FJSP best-known presets. "
            "The instance current_time and horizon are read directly from the input JSON."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input JSON path")
    parser.add_argument(
        "--track",
        choices=sorted(DEFAULT_PRESETS),
        default="relaxed",
        help="Preset track: relaxed=infinite batch capacity, finite=finite batch capacity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional solution JSON path. Defaults to outputs/<case>_<horizon>/best_<track>.json",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the preset output path without generating a new solution",
    )
    parser.add_argument("--rebuild-instance", action="store_true")
    parser.add_argument("--rebuild-setup", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    """统一把命令行相对路径解析到项目根目录下。"""
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def known_case_slug(input_path: Path) -> str:
    """把输入文件名映射为预设名；未知文件走通用预设。"""
    stem = input_path.stem
    if stem == "测试算例1" or stem == "实际规模输入数据":
        return "test_case1"
    if stem == "测试算例2" or stem == "input_data":
        return "test_case2"
    checksum = zlib.crc32(str(input_path.resolve()).encode("utf-8")) & 0xFFFFFFFF
    return f"case_{checksum:08x}"


def default_cache_paths(root: Path, input_path: Path, track: str) -> tuple[Path, Path]:
    """按输入文件和轨道拆分缓存，避免 relaxed 与 finite 互相覆盖。"""
    checksum = zlib.crc32(str(input_path.resolve()).encode("utf-8")) & 0xFFFFFFFF
    prefix = f"preset_{checksum:08x}_{track}"
    return root / "cache" / f"{prefix}_instance.pkl", root / "cache" / f"{prefix}_setup.sqlite"


def default_output_path(root: Path, input_path: Path, horizon: int, track: str) -> Path:
    """把主推输出固定到 outputs/<case>_<horizon>/best_<track>.json。"""
    case_slug = known_case_slug(input_path)
    return root / "outputs" / f"{case_slug}_{horizon}" / f"best_{track}.json"


def preset_summary(
    preset: SolverPreset,
    instance: Any,
    track: str,
    case_slug: str,
) -> dict[str, Any]:
    """只打印足以识别当前预设的关键参数。

    完整参数较多，全部打印会降低可读性。本函数保留最影响结果口径的
    字段：时间口径、是否有限组批、lookahead/start_guard、zero setup
    奖励、任务 bonus 和组批策略。
    """
    return {
        "case": case_slug,
        "track": track,
        "current_time_from_json": instance.current_time,
        "horizon_from_json": instance.horizon,
        "maintenance_shift": preset.maintenance_shift,
        "finite_batch_capacity": preset.finite_batch_capacity,
        "lookahead": preset.lookahead,
        "start_guard": preset.start_guard,
        "score_zero_setup": preset.score_zero_setup,
        "phase2_zero_setup": preset.phase2_zero_setup,
        "task_bonus": preset.task_bonus_map,
        "batch_group_wait": preset.batch_group_wait,
        "batch_group_mixed_time": preset.batch_group_mixed_time,
        "batch_group_any_time": preset.batch_group_any_time,
    }


def run_validate(
    track: str,
    preset: SolverPreset,
    instance: Any,
    setup_store: SetupRowStore,
    input_path: Path,
    output_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """根据产能口径选择对应校验器。

    relaxed 轨道只检查普通机器和任务内部约束；finite 轨道还要检查
    组批 family、容量、批时长和批组重叠。因此不能统一调用同一个
    校验函数。
    """
    if track == "finite":
        # 有限组批校验需要知道每个任务选中的路径，才能从原始 JSON
        # 恢复 family 和批容量等元数据。
        force_path_map = infer_force_path_map_from_solution(output_path)
        return validate_batch_solution(
            instance=instance,
            setup_store=setup_store,
            input_path=input_path,
            solution_path=output_path,
            force_path_map=force_path_map,
            path_nonbatch_mult=preset.path_nonbatch_mult,
            path_batch_weight=preset.path_batch_weight,
            path_wait_weight=preset.path_wait_weight,
            path_machine_penalties=None,
        )
    return validate_solution(instance, setup_store, output_path)


def main() -> int:
    """预设求解入口。

    执行顺序：
    1. 解析输入路径，并根据文件名选择算例专属预设或通用预设。
    2. 用预设中的路径选择参数构建 InstanceData。
    3. 如果不是 validate-only，就实例化调度器、生成完整解并写入文件。
    4. 调用对应校验器验证刚生成或已有的解文件。

    返回码约定：0 表示校验通过，1 表示解存在约束错误。
    """
    register_pickle_compat_aliases()
    args = parse_args()
    root = args.root.resolve()
    input_path = resolve_path(root, args.input) if args.input else detect_input_json(root)
    case_slug = known_case_slug(input_path)
    preset = CASE_PRESETS.get(case_slug, DEFAULT_PRESETS)[args.track]
    instance_cache, setup_db = default_cache_paths(root, input_path, args.track)

    print(f"[preset] input: {input_path}", flush=True)
    # 按预设的路径选择策略构建算例缓存。除非预设显式指定，
    # 否则不改动 JSON 自带的时间口径。
    instance = build_instance(
        root=root,
        input_path=input_path,
        cache_path=instance_cache,
        force=args.rebuild_instance,
        path_nonbatch_mult=preset.path_nonbatch_mult,
        path_batch_weight=preset.path_batch_weight,
        path_wait_weight=preset.path_wait_weight,
        path_machine_penalties=None,
        force_path_map=None,
        current_time_override=None,
        zero_current_time=False,
        maintenance_shift=preset.maintenance_shift,
    )
    output_path = resolve_path(root, args.output) if args.output else default_output_path(
        root, input_path, instance.horizon, args.track
    )
    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=args.rebuild_setup)

    try:
        print(
            json.dumps(
                preset_summary(preset, instance, args.track, case_slug),
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not args.validate_only:
            # 同一个调度器同时支持 relaxed 和 finite 两种轨道；
            # 其中 finite 标志只改变组批工序是否占用有限机器时间线。
            scheduler = RelaxedRLScheduler(
                instance,
                setup_store,
                lookahead=preset.lookahead,
                start_guard=preset.start_guard,
                score_weight=preset.score_weight,
                score_density=preset.score_density,
                score_started=preset.score_started,
                score_family=preset.score_family,
                score_progress=preset.score_progress,
                score_zero_setup=preset.score_zero_setup,
                score_setup_fixed=preset.score_setup_fixed,
                score_setup_per=preset.score_setup_per,
                score_est_final_per=preset.score_est_final_per,
                task_bonus_map=preset.task_bonus_map,
                defer_task_ids=preset.defer_task_ids,
                force_machine_map=None,
                phase2_started=preset.phase2_started,
                phase2_density=preset.phase2_density,
                phase2_family=preset.phase2_family,
                phase2_progress=preset.phase2_progress,
                phase2_zero_setup=preset.phase2_zero_setup,
                phase2_setup_fixed=preset.phase2_setup_fixed,
                phase2_setup_per=preset.phase2_setup_per,
                phase2_finish_per=preset.phase2_finish_per,
                phase2_started_gate=not preset.phase2_allow_unstarted,
                finite_batch_capacity=preset.finite_batch_capacity,
                batch_group_wait=preset.batch_group_wait,
                batch_group_mixed_time=preset.batch_group_mixed_time,
                batch_group_any_time=preset.batch_group_any_time,
            )
            task_records = scheduler.solve()
            dump_solution(instance, task_records, output_path)
            print(json.dumps(scheduler.metrics(), ensure_ascii=False, indent=2), flush=True)

        # 始终校验落盘文件，包括刚刚生成的新解。
        errors, metrics = run_validate(
            args.track,
            preset,
            instance,
            setup_store,
            input_path,
            output_path,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if errors:
            print("[preset-validate] sample errors:", flush=True)
            for item in errors[:50]:
                print(item, flush=True)
            return 1
        print(f"[preset-validate] no errors; output: {output_path}", flush=True)
        return 0
    finally:
        setup_store.close()


if __name__ == "__main__":
    sys.exit(main())

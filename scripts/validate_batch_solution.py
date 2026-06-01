#!/usr/bin/env python3
"""有限组批口径解文件校验入口。

本文件在 relaxed 校验口径基础上额外检查 p-batch 约束：同一批 family
一致、批大小不超过容量、批时长等于批内最大单件加工时长、同一机器批组
之间不重叠。finite 轨道输出应使用本脚本校验。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ijson

from rl_relaxed_solver import (
    INF,
    ScheduledOp,
    SetupRowStore,
    build_instance,
    choose_path,
    detect_input_json,
    fmt_dt,
    infer_force_path_map_from_solution,
    load_solution_records,
    parse_named_float_map,
    parse_named_str_map,
)


@dataclass(frozen=True)
class BatchMeta:
    """从原始输入 JSON 恢复的静态组批属性。"""

    family: tuple[str, ...]
    capacity: int


@dataclass(frozen=True)
class BatchItem:
    """时间戳转为分钟轴后的单个已排组批工序。"""

    task_id: str
    seq: str
    proc_id: str
    process_time: int
    family: tuple[str, ...]
    capacity: int
    start: int
    finish: int


def resolve_path(root: Path, path: Path) -> Path:
    """把命令行相对路径解析到项目根目录下。"""
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def time_label(instance: Any, value: int) -> str:
    """在错误信息中同时展示分钟值和日历时间。"""
    return f"{value} ({fmt_dt(instance.start_dt, instance.current_time, value)})"


def as_minutes(instance: Any, record: ScheduledOp) -> tuple[int, int]:
    """把输出文件中的日历时间转回算例分钟轴。"""
    start = instance.current_time + int((record.start - instance.start_dt).total_seconds() // 60)
    finish = instance.current_time + int((record.finish - instance.start_dt).total_seconds() // 60)
    return start, finish


def load_batch_metadata(
    input_path: Path,
    force_path_map: dict[str, str],
    path_nonbatch_mult: float,
    path_batch_weight: float,
    path_wait_weight: float,
    path_machine_penalties: Optional[dict[str, float]],
) -> dict[tuple[str, str, str, str], BatchMeta]:
    """加载普通紧凑算例缓存中没有保留的 family/容量数据。

    调度器的紧凑缓存只保存派工决策需要的字段。完整有限组批校验还需要
    每个任务、路径、机器组合上的 family 和批容量，所以这里重新流式读取
    原始 JSON，只为已选路径恢复这部分元数据。
    """

    metadata: dict[tuple[str, str, str, str], BatchMeta] = {}
    with input_path.open("rb") as fh:
        for task_id, task_payload in ijson.kvitems(fh, "task"):
            task_id = str(task_id)
            path_id = choose_path(
                task_id,
                task_payload,
                path_nonbatch_mult=path_nonbatch_mult,
                path_batch_weight=path_batch_weight,
                path_wait_weight=path_wait_weight,
                path_machine_penalties=path_machine_penalties,
                force_path_map=force_path_map,
            )
            path_payload = task_payload["process_path"][path_id]
            for seq, proc_payload in path_payload["process_list"].items():
                if not bool(proc_payload.get("is_batch_type")):
                    continue
                family = tuple(str(item) for item in proc_payload.get("batch_family", ()))
                for machine_id, eqp_payload in proc_payload["eqp_list"].items():
                    raw_capacity = eqp_payload.get("curr_batch_size", 1)
                    capacity = max(1, int(float(raw_capacity)))
                    metadata[(task_id, str(path_id), str(seq), str(machine_id))] = BatchMeta(
                        family=family,
                        capacity=capacity,
                    )
    return metadata


def validate_batch_solution(
    instance: Any,
    setup_store: SetupRowStore,
    input_path: Path,
    solution_path: Path,
    force_path_map: dict[str, str],
    path_nonbatch_mult: float,
    path_batch_weight: float,
    path_wait_weight: float,
    path_machine_penalties: Optional[dict[str, float]],
) -> tuple[list[str], dict[str, Any]]:
    """按有限 p-batch 产能规则校验一个解。

    这个函数会先做 relaxed 校验中也需要的任务内部检查，再额外收集所有
    组批工序。普通机器按机器时间线检查 setup 和重叠；组批机器按批组检查
    family、容量、批时长和批组之间的重叠。
    """
    raw_records = load_solution_records(solution_path)
    batch_meta = load_batch_metadata(
        input_path=input_path,
        force_path_map=force_path_map,
        path_nonbatch_mult=path_nonbatch_mult,
        path_batch_weight=path_batch_weight,
        path_wait_weight=path_wait_weight,
        path_machine_penalties=path_machine_penalties,
    )

    errors: list[str] = []
    nonbatch_by_machine: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    batch_groups: dict[tuple[str, int, int], list[BatchItem]] = defaultdict(list)
    completed_weight = 0.0
    completed_tasks = 0
    setup_count = 0
    fully_scheduled_tasks = 0
    valid_full_tasks = 0
    scheduled_ops = 0
    batch_ops = 0

    missing_tasks = sorted(set(instance.tasks) - set(raw_records))
    if missing_tasks:
        sample = ", ".join(missing_tasks[:10])
        errors.append(f"output misses {len(missing_tasks)} tasks; sample=[{sample}]")

    # 第一遍：校验任务内部记录，并收集机器时间线。
    # 有限组批调度器会让同一批共享 (machine, start, finish)，
    # 因此这里也按这三个字段聚合批组。
    for task_id, parsed in raw_records.items():
        if task_id not in instance.tasks:
            errors.append(f"{task_id}: unknown task in output")
            continue

        task = instance.tasks[task_id]
        expected_seqs = list(task.process_order)
        actual_seqs = [record.seq for record in parsed]
        if actual_seqs != expected_seqs:
            errors.append(
                f"{task_id}: process seqs do not cover selected path; "
                f"expected={expected_seqs}, actual={actual_seqs}, path_id={task.path_id}"
            )
            continue

        fully_scheduled_tasks += 1
        task_has_errors = False
        proc_records: dict[str, ScheduledOp] = {}

        for idx, record in enumerate(parsed):
            proc = task.processes[idx]
            if record.path_id != task.path_id:
                errors.append(
                    f"{task_id}:{record.seq}: path_id mismatch; "
                    f"expected={task.path_id}, actual={record.path_id}"
                )
                task_has_errors = True
                continue

            candidate = next(
                (item for item in proc.candidates if item.machine_id == record.machine_id),
                None,
            )
            if candidate is None:
                allowed = [item.machine_id for item in proc.candidates]
                errors.append(
                    f"{task_id}:{record.seq}: invalid machine {record.machine_id}; "
                    f"allowed={allowed}"
                )
                task_has_errors = True
                continue

            if record.machine_id not in instance.machines:
                errors.append(f"{task_id}:{record.seq}: unknown machine {record.machine_id}")
                task_has_errors = True
                continue

            start_min, finish_min = as_minutes(instance, record)
            duration = finish_min - start_min
            if duration <= 0:
                errors.append(
                    f"{task_id}:{record.seq}: non-positive duration; "
                    f"start={time_label(instance, start_min)}, finish={time_label(instance, finish_min)}"
                )
                task_has_errors = True
                continue

            if proc.is_batch:
                # 对组批工序来说，记录时长可以大于单件加工时长，
                # 因为整批加工时长取批内最大加工时长。
                if duration < candidate.process_time:
                    errors.append(
                        f"{task_id}:{record.seq}: batch duration shorter than process time; "
                        f"actual={duration}, minimum={candidate.process_time}, machine={record.machine_id}"
                    )
                    task_has_errors = True
                    continue
            elif duration != candidate.process_time:
                errors.append(
                    f"{task_id}:{record.seq}: duration mismatch; "
                    f"actual={duration}, expected={candidate.process_time}, machine={record.machine_id}"
                )
                task_has_errors = True
                continue

            overlaps_down = False
            for down_start, down_end in instance.machines[record.machine_id].down_intervals:
                if not (finish_min <= down_start or start_min >= down_end):
                    errors.append(
                        f"{task_id}:{record.seq}: overlaps maintenance on {record.machine_id}; "
                        f"op=[{time_label(instance, start_min)}, {time_label(instance, finish_min)}), "
                        f"down=[{time_label(instance, down_start)}, {time_label(instance, down_end)})"
                    )
                    overlaps_down = True
                    task_has_errors = True
                    break
            if overlaps_down:
                continue

            scheduled = ScheduledOp(
                task_id=record.task_id,
                seq=record.seq,
                path_id=record.path_id,
                machine_id=record.machine_id,
                start=start_min,
                finish=finish_min,
            )
            proc_records[proc.proc_id] = scheduled
            scheduled_ops += 1
            if proc.is_batch:
                batch_ops += 1
                meta_key = (task_id, task.path_id, proc.seq, record.machine_id)
                meta = batch_meta.get(meta_key)
                if meta is None:
                    errors.append(
                        f"{task_id}:{record.seq}: missing batch metadata for machine "
                        f"{record.machine_id}, path={task.path_id}"
                    )
                    task_has_errors = True
                    continue
                batch_groups[(record.machine_id, start_min, finish_min)].append(
                    BatchItem(
                        task_id=task_id,
                        seq=record.seq,
                        proc_id=proc.proc_id,
                        process_time=candidate.process_time,
                        family=meta.family,
                        capacity=meta.capacity,
                        start=start_min,
                        finish=finish_min,
                    )
                )
            else:
                nonbatch_by_machine[record.machine_id].append(
                    (start_min, finish_min, task_id, proc.proc_id)
                )

        # 第二遍任务内部校验：在已知该任务所有有效记录后，
        # 统一计算释放、前序、转运和 q-time 边界。
        for idx, record in enumerate(parsed):
            proc = task.processes[idx]
            proc_record = proc_records.get(proc.proc_id)
            if proc_record is None:
                continue

            lower = max(instance.current_time, task.earliest_ava_time)
            upper = INF
            if idx > 0:
                prev_proc = task.processes[idx - 1]
                prev_record = proc_records.get(prev_proc.proc_id)
                if prev_record is None:
                    errors.append(
                        f"{task_id}:{record.seq}: missing valid predecessor record "
                        f"for seq={prev_proc.seq}"
                    )
                    task_has_errors = True
                    continue

                from_factory = instance.machines[prev_record.machine_id].factory
                to_factory = instance.machines[proc_record.machine_id].factory
                if from_factory != to_factory and (from_factory, to_factory) not in prev_proc.diff_factory_info:
                    errors.append(
                        f"{task_id}:{record.seq}: illegal factory transfer; "
                        f"{prev_record.machine_id}({from_factory}) -> "
                        f"{proc_record.machine_id}({to_factory})"
                    )
                    task_has_errors = True

                lower = max(
                    lower,
                    prev_record.finish
                    + instance.transitions.get(prev_record.machine_id, {}).get(
                        proc_record.machine_id, 0
                    ),
                )

            for qtime in task.incoming_qtimes[idx]:
                anchor_idx = task.seq_to_idx.get(qtime.start_seq)
                if anchor_idx is None:
                    errors.append(
                        f"{task_id}:{record.seq}: q-time anchor seq not in selected path; "
                        f"anchor={qtime.start_seq}"
                    )
                    task_has_errors = True
                    continue
                anchor_proc = task.processes[anchor_idx]
                anchor_record = proc_records.get(anchor_proc.proc_id)
                if anchor_record is None:
                    errors.append(
                        f"{task_id}:{record.seq}: missing q-time anchor record; "
                        f"anchor_seq={qtime.start_seq}"
                    )
                    task_has_errors = True
                    continue
                anchor = anchor_record.start if qtime.start_type == "start" else anchor_record.finish
                offset = proc_record.finish - proc_record.start if qtime.end_type == "end" else 0
                if qtime.min_interval is not None:
                    lower = max(lower, anchor + qtime.min_interval - offset)
                if qtime.max_interval is not None:
                    upper = min(upper, anchor + qtime.max_interval - offset)

            if proc_record.start < lower:
                errors.append(
                    f"{task_id}:{record.seq}: violates lower time bound; "
                    f"start={time_label(instance, proc_record.start)}, "
                    f"lower={time_label(instance, lower)}"
                )
                task_has_errors = True
            if proc_record.start > upper:
                errors.append(
                    f"{task_id}:{record.seq}: violates upper time bound; "
                    f"start={time_label(instance, proc_record.start)}, "
                    f"upper={time_label(instance, upper)}"
                )
                task_has_errors = True

        if len(proc_records) == len(task.processes):
            valid_full_tasks += 1
            last_proc_id = task.processes[-1].proc_id
            last = proc_records.get(last_proc_id)
            if last is not None and not task_has_errors and last.finish <= instance.horizon:
                completed_tasks += 1
                completed_weight += task.weight

    for machine_id, entries in nonbatch_by_machine.items():
        entries.sort()
        prev_finish: Optional[int] = None
        prev_proc_id: Optional[str] = None
        for start, _finish, task_id, proc_id in entries:
            if prev_finish is not None and prev_proc_id is not None:
                setup_time = setup_store.get(prev_proc_id, proc_id)
                if setup_time > 0:
                    setup_count += 1
                if start < prev_finish + setup_time:
                    errors.append(
                        f"{machine_id}: non-batch overlap/setup violation; "
                        f"prev_proc={prev_proc_id}, next_proc={proc_id}, "
                        f"prev_finish={time_label(instance, prev_finish)}, setup={setup_time}, "
                        f"next_start={time_label(instance, start)}, next_task={task_id}"
                    )
            prev_finish = _finish
            prev_proc_id = proc_id

    # 有限组批校验故意和普通机器校验分开。一个批组合法需要满足：
    # 批内 family 一致、批大小不超过最紧容量、批时长等于单件最大加工时长。
    batch_groups_by_machine: dict[str, list[tuple[int, int, tuple[str, int, int]]]] = defaultdict(list)
    for group_key, items in batch_groups.items():
        machine_id, start, finish = group_key
        duration = finish - start
        families = {item.family for item in items}
        capacities = [item.capacity for item in items]
        max_process_time = max(item.process_time for item in items)
        min_capacity = min(capacities) if capacities else 1

        if len(families) > 1:
            detail = sorted(
                f"{item.task_id}:{item.seq}:{'/'.join(item.family)}" for item in items
            )
            errors.append(
                f"{machine_id}: batch family mismatch in group "
                f"[{time_label(instance, start)}, {time_label(instance, finish)}); items={detail}"
            )
        if len(items) > min_capacity:
            detail = [f"{item.task_id}:{item.seq}" for item in items]
            errors.append(
                f"{machine_id}: batch capacity exceeded; group_size={len(items)}, "
                f"capacity={min_capacity}, items={detail}"
            )
        if duration != max_process_time:
            detail = [
                f"{item.task_id}:{item.seq}:ptime={item.process_time}" for item in items
            ]
            errors.append(
                f"{machine_id}: batch duration mismatch; actual={duration}, "
                f"expected_max_process_time={max_process_time}, "
                f"group=[{time_label(instance, start)}, {time_label(instance, finish)}), "
                f"items={detail}"
            )
        batch_groups_by_machine[machine_id].append((start, finish, group_key))

    for machine_id, groups in batch_groups_by_machine.items():
        groups.sort()
        prev_finish: Optional[int] = None
        prev_key: Optional[tuple[str, int, int]] = None
        for start, finish, group_key in groups:
            if prev_finish is not None and prev_key is not None and start < prev_finish:
                errors.append(
                    f"{machine_id}: overlapping batch groups; "
                    f"prev=[{time_label(instance, prev_key[1])}, {time_label(instance, prev_key[2])}), "
                    f"next=[{time_label(instance, start)}, {time_label(instance, finish)})"
                )
            prev_finish = finish
            prev_key = group_key

    metrics = {
        "completed_tasks_within_horizon": completed_tasks,
        "completed_weight_within_horizon": round(completed_weight, 3),
        "setup_count_positive": setup_count,
        "fully_scheduled_tasks": fully_scheduled_tasks,
        "valid_full_tasks": valid_full_tasks,
        "total_tasks": len(instance.tasks),
        "scheduled_ops": scheduled_ops,
        "batch_ops": batch_ops,
        "batch_group_count": len(batch_groups),
        "machine_count_with_nonbatch_load": len(nonbatch_by_machine),
        "machine_count_with_batch_load": len(batch_groups_by_machine),
        "error_count": len(errors),
    }
    return errors, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Huawei FJSP solution JSON with finite p-batch machine "
            "grouping constraints. This script is independent from the original "
            "relaxed validator."
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
        "--current-time-override",
        type=int,
        default=None,
        help="Override time.current_time without editing the input JSON.",
    )
    parser.add_argument(
        "--maintenance-shift",
        type=int,
        default=None,
        help="Shift all eqp_down_interval bounds by this many minutes.",
    )
    parser.add_argument(
        "--zero-current-time",
        action="store_true",
        help=(
            "Treat the input current moment as minute 0. If --maintenance-shift "
            "is omitted, shift maintenance by the original input current_time."
        ),
    )
    parser.add_argument(
        "--instance-cache",
        type=Path,
        default=Path("cache/batch_validation_instance.pkl"),
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
    1. 解析输入算例、有限组批解文件、缓存路径和路径选择参数。
    2. 从解文件推断任务选路，保证恢复的 family/容量元数据与解一致。
    3. 构建 InstanceData，并按需要覆盖 horizon 或旧实验时间口径。
    4. 加载 setup 行缓存，执行有限组批校验。
    5. 打印指标、错误明细，并在需要时写出 JSON 报告。

    返回码约定：0 表示校验通过，1 表示发现约束错误，2 表示输入文件缺失。
    """
    args = parse_args()
    root = args.root.resolve()
    input_path = args.input.resolve() if args.input else detect_input_json(root)
    solution_path = resolve_path(root, args.solution)
    instance_cache = resolve_path(root, args.instance_cache)
    setup_db = resolve_path(root, args.setup_db)

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
        # 输出文件中每道工序都带有 path_id，可直接反推出任务选路，
        # 避免要求用户手动复制很长的 --force-path 参数。
        force_path_map = infer_force_path_map_from_solution(solution_path)
        inferred_force_paths = bool(force_path_map)

    print(f"[batch-validate] input: {input_path}", flush=True)
    print(f"[batch-validate] solution: {solution_path}", flush=True)
    if inferred_force_paths:
        print(
            f"[batch-validate] inferred force paths from solution: {len(force_path_map)}",
            flush=True,
        )

    maintenance_shift = args.maintenance_shift
    if args.zero_current_time and maintenance_shift is None:
        # 早期实验曾把输入 current_time 重新解释为分钟 0。
        # 这个兼容模式只用于审计旧结果。
        with input_path.open("rb") as fh:
            maintenance_shift = int(next(ijson.items(fh, "time.current_time")))
    elif maintenance_shift is None:
        maintenance_shift = 0

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
        current_time_override=args.current_time_override,
        zero_current_time=args.zero_current_time,
        maintenance_shift=maintenance_shift,
    )
    if args.zero_current_time or args.current_time_override is not None or maintenance_shift:
        print(
            json.dumps(
                {
                    "effective_current_time": instance.current_time,
                    "maintenance_shift": maintenance_shift,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.horizon_override is not None:
        instance.horizon = int(args.horizon_override)
        print(f"[batch-validate] horizon override: {instance.horizon}", flush=True)

    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=args.rebuild_setup)
    try:
        errors, metrics = validate_batch_solution(
            instance=instance,
            setup_store=setup_store,
            input_path=input_path,
            solution_path=solution_path,
            force_path_map=force_path_map,
            path_nonbatch_mult=args.path_nonbatch_mult,
            path_batch_weight=args.path_batch_weight,
            path_wait_weight=args.path_wait_weight,
            path_machine_penalties=path_machine_penalties,
        )
    finally:
        setup_store.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    shown_errors = errors if args.max_errors < 0 else errors[: args.max_errors]
    if errors:
        print(f"[batch-validate] errors: {len(errors)}", flush=True)
        if args.max_errors >= 0 and len(errors) > len(shown_errors):
            print(
                f"[batch-validate] showing first {len(shown_errors)} errors; "
                "rerun with --max-errors -1 to print all",
                flush=True,
            )
        for idx, item in enumerate(shown_errors, start=1):
            print(f"[{idx:04d}] {item}", flush=True)
    else:
        print("[batch-validate] no errors", flush=True)

    if args.report is not None:
        report_path = resolve_path(root, args.report)
        report: dict[str, Any] = {
            "input": str(input_path),
            "solution": str(solution_path),
            "horizon": instance.horizon,
            "validator": "validate_batch_solution",
            "batch_capacity_rule": "same machine/start/finish forms one batch; same family; size <= curr_batch_size; duration == max process_time",
            "inferred_force_path_count": len(force_path_map) if inferred_force_paths else 0,
            "metrics": metrics,
            "errors": shown_errors,
            "error_count": len(errors),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"[batch-validate] report: {report_path}", flush=True)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

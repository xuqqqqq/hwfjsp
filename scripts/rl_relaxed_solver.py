#!/usr/bin/env python3
"""核心求解器和 relaxed 口径校验逻辑。

本文件负责三件事：
1. 读取原始算例 JSON，并转换为求解器使用的紧凑数据结构。
2. 用两阶段启发式调度器生成完整解。
3. 在“组批机器无限产能”口径下校验解是否合法。

时间口径：内部统一使用分钟轴；写出解文件时再转换成日历时间字符串。
维修窗口：内部按半开区间 [start, end) 判断冲突。
setup 统计：只统计普通机器上 setup_time > 0 的相邻工序切换。
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sqlite3
import sys
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import ijson


DATE_FORMATS = ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")
INF = 10**18


def to_int(value: Any) -> Optional[int]:
    """把 JSON 里的数值字段转成 int；None 保持为 None。"""
    if value is None:
        return None
    return int(value)


def to_float(value: Any) -> float:
    """把 JSON 里的数值字段转成 float。"""
    return float(value)


def parse_dt(value: str) -> datetime:
    """兼容两种常见日期格式，统一转成 datetime。"""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime: {value}")


def fmt_dt(start_dt: datetime, current_time: int, value: int) -> str:
    """把内部分钟轴时间转换为输出 JSON 需要的日期字符串。"""
    return (start_dt + timedelta(minutes=value - current_time)).strftime("%Y/%m/%d %H:%M:%S")


def normalize_input_path(path: Path) -> str:
    """把输入路径标准化，作为缓存签名的一部分。"""
    return str(path.resolve())


def input_signature(path: Path) -> tuple[str, int, int]:
    """记录输入文件路径、大小和修改时间，用来判断缓存是否失效。"""
    stat = path.stat()
    return (normalize_input_path(path), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class CandidateSpec:
    """选定路径下某道工序的一个候选机器。

    一道工序通常可以在多台机器上加工，每台机器对应不同加工时长、
    优先级和批容量。调度时的“动作”本质上就是从这些候选机器里选一个。
    """

    machine_id: str
    process_time: int
    priority: float
    batch_size: int


@dataclass(frozen=True)
class QTimeSpec:
    """两道工序事件之间的最小/最大间隔约束。

    start_type/end_type 决定约束锚点是工序开始还是结束。例如“上一道结束到
    下一道开始至少等待 30 分钟”会转成一个 lower bound。
    """

    start_seq: str
    start_type: str
    end_seq: str
    end_type: str
    min_interval: Optional[int]
    max_interval: Optional[int]


@dataclass
class ProcessSpec:
    """调度器使用的紧凑工序表示。

    原始 JSON 的工序字段较多，本结构仅保留求解和校验所需字段：
    工序编号、是否组批、跨厂可转运关系、候选机器和最短加工时间。
    """

    seq: str
    proc_id: str
    is_batch: bool
    batch_family: tuple[str, ...]
    diff_factory_info: tuple[tuple[str, str], ...]
    candidates: tuple[CandidateSpec, ...]
    min_process_time: int


@dataclass
class TaskSpec:
    """已选路径下的工件模型，包含预计算的乐观尾部时间。

    每个任务在 build_instance() 阶段已经选定一条加工路径，所以调度器不再
    同时处理多条路径。optimistic_total_from / optimistic_nonbatch_from 用于
    快速估计“如果现在排这道工序，后面最理想还需要多久”。
    """

    task_id: str
    earliest_ava_time: int
    delivery_time: int
    priority: float
    weight: float
    path_id: str
    process_order: tuple[str, ...]
    seq_to_idx: dict[str, int]
    processes: tuple[ProcessSpec, ...]
    incoming_qtimes: tuple[tuple[QTimeSpec, ...], ...]
    optimistic_total_from: tuple[int, ...]
    optimistic_nonbatch_from: tuple[int, ...]


@dataclass
class MachineSpec:
    """机器日历与所属工厂信息。"""

    machine_id: str
    factory: str
    down_intervals: tuple[tuple[int, int], ...]


@dataclass
class ScheduledOp:
    """分钟轴上的一条已排工序记录。"""

    task_id: str
    seq: str
    path_id: str
    machine_id: str
    start: int
    finish: int


@dataclass
class CandidateEval:
    """一个可行动作及其评分策略所需的派生特征。

    evaluate_candidate() 会把硬约束都检查完，能生成 CandidateEval 就说明
    该动作当前可行。score_candidate() 和 score_candidate_phase2() 只负责
    在可行动作之间做偏好排序。
    """

    task_id: str
    idx: int
    machine_id: str
    start: int
    finish: int
    setup_time: int
    est_final: int
    upper_bound: int
    option_priority: float
    started: bool
    same_family: bool
    zero_setup: bool


@dataclass
class InstanceData:
    """求解和校验共用的解析后算例缓存。"""

    source_input: str
    source_size: int
    source_mtime_ns: int
    current_time: int
    current_dt: str
    start_dt: datetime
    horizon: int
    machines: dict[str, MachineSpec]
    transitions: dict[str, dict[str, int]]
    tasks: dict[str, TaskSpec]
    path_strategy_key: str = "default"


def parse_named_float_map(raw: str) -> dict[str, float]:
    """解析 name=value,name=value 形式的浮点参数。"""
    result: dict[str, float] = {}
    text = raw.strip()
    if not text:
        return result
    for item in text.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected name=value pair, got: {chunk}")
        name, value = chunk.split("=", 1)
        result[name.strip()] = float(value.strip())
    return result


def parse_named_str_map(raw: str) -> dict[str, str]:
    """解析 name=value,name=value 形式的字符串参数。"""
    result: dict[str, str] = {}
    text = raw.strip()
    if not text:
        return result
    for item in text.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected name=value pair, got: {chunk}")
        name, value = chunk.split("=", 1)
        result[name.strip()] = value.strip()
    return result


def parse_force_machine_map(raw: str) -> dict[tuple[str, str], str]:
    """解析 task_id:seq=machine_id 形式的强制机器参数。"""
    result: dict[tuple[str, str], str] = {}
    text = raw.strip()
    if not text:
        return result
    for item in text.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk or ":" not in chunk.split("=", 1)[0]:
            raise ValueError(f"Expected task_id:seq=machine_id pair, got: {chunk}")
        lhs, machine_id = chunk.split("=", 1)
        task_id, seq = lhs.split(":", 1)
        task_id = task_id.strip()
        seq = seq.strip()
        machine_id = machine_id.strip()
        if not task_id or not seq or not machine_id:
            raise ValueError(f"Expected task_id:seq=machine_id pair, got: {chunk}")
        result[(task_id, seq)] = machine_id
    return result


def parse_name_set(raw: str) -> set[str]:
    """解析逗号分隔的名称集合。"""
    text = raw.strip()
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def infer_force_path_map_from_solution(output_path: Path) -> dict[str, str]:
    """从已有解文件中恢复每个任务选择的加工路径。"""
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    result: dict[str, str] = {}
    for task_id, task_payload in payload.get("task", {}).items():
        path_ids = {
            str(row["path_id"])
            for row in task_payload.get("process_path", {}).values()
            if "path_id" in row
        }
        if len(path_ids) == 1:
            result[str(task_id)] = next(iter(path_ids))
    return result


def path_strategy_signature(
    path_nonbatch_mult: float,
    path_batch_weight: float,
    path_wait_weight: float,
    path_machine_penalties: dict[str, float],
    force_path_map: dict[str, str],
    current_time_override: Optional[int],
    zero_current_time: bool,
    maintenance_shift: Optional[int],
) -> str:
    """为会影响算例解析结果的路径/时间策略生成缓存键。"""
    payload = {
        "path_nonbatch_mult": round(path_nonbatch_mult, 6),
        "path_batch_weight": round(path_batch_weight, 6),
        "path_wait_weight": round(path_wait_weight, 6),
        "path_machine_penalties": dict(sorted(path_machine_penalties.items())),
        "force_path_map": dict(sorted(force_path_map.items())),
        "current_time_override": current_time_override,
        "zero_current_time": zero_current_time,
        "maintenance_shift": maintenance_shift,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def register_pickle_compat_aliases() -> None:
    """兼容早期以 __main__ 执行本文件时生成的 pickle 缓存。"""
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    for symbol in (
        CandidateSpec,
        QTimeSpec,
        ProcessSpec,
        TaskSpec,
        MachineSpec,
        ScheduledOp,
        CandidateEval,
        InstanceData,
    ):
        if not hasattr(main_mod, symbol.__name__):
            setattr(main_mod, symbol.__name__, symbol)


class SetupRowStore:
    """基于 SQLite 的稀疏 setup 矩阵，并带一个小型行级 LRU 缓存。"""

    def __init__(self, db_path: Path, row_cache_size: int = 256) -> None:
        self.db_path = db_path
        self.row_cache_size = row_cache_size
        self.conn: Optional[sqlite3.Connection] = None
        self.row_cache: OrderedDict[str, dict[str, int]] = OrderedDict()

    def connect(self) -> sqlite3.Connection:
        """惰性打开 SQLite 连接，避免在对象构造阶段占用文件句柄。"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def close(self) -> None:
        """关闭 SQLite 连接；主流程 finally 中会调用。"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _remove_db_files(self) -> None:
        """删除 SQLite 主文件和 WAL/SHM 辅助文件。"""
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(self.db_path) + suffix)
            if target.exists():
                target.unlink()

    def _db_matches_input(self, input_path: Path) -> bool:
        """检查 setup 缓存是否来自当前输入文件。"""
        if not self.db_path.exists():
            return False
        expected_path, expected_size, expected_mtime = input_signature(input_path)
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.Error:
            conn.close()
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass
        meta = {key: value for key, value in rows}
        return (
            meta.get("source_input") == expected_path
            and meta.get("source_size") == str(expected_size)
            and meta.get("source_mtime_ns") == str(expected_mtime)
        )

    def ensure(self, input_path: Path, force: bool = False) -> None:
        """为当前输入文件创建或复用 setup 行数据库。"""
        if force:
            self.close()
            self._remove_db_files()
        elif self.db_path.exists():
            if self._db_matches_input(input_path):
                return
            self.close()
            print("[setup] cache mismatch detected, rebuilding", flush=True)
            self._remove_db_files()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                "CREATE TABLE setup_rows (from_proc TEXT PRIMARY KEY, payload BLOB NOT NULL)"
            )
            sig_path, sig_size, sig_mtime = input_signature(input_path)
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [
                    ("source_input", sig_path),
                    ("source_size", str(sig_size)),
                    ("source_mtime_ns", str(sig_mtime)),
                ],
            )
            batch: list[tuple[str, bytes]] = []
            row_count = 0
            with input_path.open("rb") as fh:
                for from_proc, row in ijson.kvitems(fh, "setup"):
                    payload = zlib.compress(
                        pickle.dumps({k: int(v) for k, v in row.items()}, protocol=4),
                        level=3,
                    )
                    batch.append((from_proc, payload))
                    row_count += 1
                    if len(batch) >= 128:
                        conn.executemany(
                            "INSERT INTO setup_rows(from_proc, payload) VALUES (?, ?)", batch
                        )
                        conn.commit()
                        batch.clear()
                        if row_count % 1024 == 0:
                            print(f"[setup] indexed rows: {row_count}", flush=True)
                if batch:
                    conn.executemany(
                        "INSERT INTO setup_rows(from_proc, payload) VALUES (?, ?)", batch
                    )
                    conn.commit()
            print(f"[setup] build complete: {row_count} rows", flush=True)
        finally:
            conn.close()

    def _load_row(self, from_proc: str) -> dict[str, int]:
        """按 from_proc 读取一整行 setup 数据，并放入 LRU 缓存。"""
        row = self.row_cache.get(from_proc)
        if row is not None:
            self.row_cache.move_to_end(from_proc)
            return row
        payload = self.connect().execute(
            "SELECT payload FROM setup_rows WHERE from_proc = ?",
            (from_proc,),
        ).fetchone()
        row = {} if payload is None else pickle.loads(zlib.decompress(payload[0]))
        self.row_cache[from_proc] = row
        if len(self.row_cache) > self.row_cache_size:
            self.row_cache.popitem(last=False)
        return row

    def get(self, from_proc: Optional[str], to_proc: str) -> int:
        """查询两个工序之间的 setup 时间；没有前序时 setup 为 0。"""
        if not from_proc:
            return 0
        return self._load_row(from_proc).get(to_proc, 0)


def detect_input_json(root: Path) -> Path:
    """当未传入 --input 时，在 data/data1 下寻找默认 JSON。"""
    data_dir = root / "data" / "data1"
    preferred_names = (
        "实际规模输入数据.json",
        "input_data.json",
        "小规模输入数据示例.json",
    )
    for name in preferred_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return max(data_dir.glob("*.json"), key=lambda p: p.stat().st_size)


def choose_path(
    task_id: str,
    task_payload: dict[str, Any],
    path_nonbatch_mult: float = 3.0,
    path_batch_weight: float = 1.0,
    path_wait_weight: float = 1.0,
    path_machine_penalties: Optional[dict[str, float]] = None,
    force_path_map: Optional[dict[str, str]] = None,
) -> str:
    """在派工前为任务选择一条加工路径。

    路径选择会最小化一个未来负载代理值：普通工序时间、组批工序时间、
    q-time 最小等待和可选机器惩罚。手动 force-path 会优先生效，
    这样历史 Pareto 实验可以被复现。
    """
    if force_path_map and task_id in force_path_map:
        forced = force_path_map[task_id]
        if forced not in task_payload["process_path"]:
            raise ValueError(f"Forced path {forced} not found for task {task_id}")
        return forced
    best_key: Optional[tuple[float, float, str]] = None
    best_path_id = ""
    machine_penalties = path_machine_penalties or {}
    for path_id, path in task_payload["process_path"].items():
        non_batch = 0.0
        batch = 0.0
        min_wait = 0.0
        machine_penalty = 0.0
        for proc in path["process_list"].values():
            min_pt = min(to_int(info["process_time"]) for info in proc["eqp_list"].values())
            if proc["is_batch_type"]:
                batch += min_pt
            else:
                non_batch += min_pt
            if machine_penalties:
                machine_penalty += min(
                    machine_penalties.get(machine_id, 0.0)
                    for machine_id in proc["eqp_list"].keys()
                )
        for qtime in path.get("qtime_info", {}).values():
            if qtime["min_process_interval"] is not None:
                min_wait += float(qtime["min_process_interval"])
        weighted_primary = (
            non_batch * path_nonbatch_mult
            + batch * path_batch_weight
            + min_wait * path_wait_weight
            + machine_penalty
        )
        key = (
            weighted_primary,
            non_batch + batch + min_wait + machine_penalty,
            path_id,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_path_id = path_id
    if not best_path_id:
        raise ValueError(f"No path found for task {task_id}")
    return best_path_id


def build_task_spec(
    task_id: str,
    task_payload: dict[str, Any],
    path_nonbatch_mult: float = 3.0,
    path_batch_weight: float = 1.0,
    path_wait_weight: float = 1.0,
    path_machine_penalties: Optional[dict[str, float]] = None,
    force_path_map: Optional[dict[str, str]] = None,
) -> TaskSpec:
    """把一个原始任务 payload 转成已选路径下的紧凑任务模型。"""
    path_id = choose_path(
        task_id,
        task_payload,
        path_nonbatch_mult=path_nonbatch_mult,
        path_batch_weight=path_batch_weight,
        path_wait_weight=path_wait_weight,
        path_machine_penalties=path_machine_penalties,
        force_path_map=force_path_map,
    )
    path = task_payload["process_path"][path_id]
    process_order = tuple(sorted(path["process_list"].keys(), key=lambda x: int(x)))
    seq_to_idx = {seq: idx for idx, seq in enumerate(process_order)}
    processes: list[ProcessSpec] = []
    incoming_qtimes: list[list[QTimeSpec]] = [[] for _ in process_order]

    sequential_waits = [0 for _ in process_order]
    for qinfo in path.get("qtime_info", {}).values():
        qspec = QTimeSpec(
            start_seq=str(qinfo["start_process_seq"]),
            start_type=str(qinfo["start_process_type"]),
            end_seq=str(qinfo["end_process_seq"]),
            end_type=str(qinfo["end_process_type"]),
            min_interval=to_int(qinfo["min_process_interval"]),
            max_interval=to_int(qinfo["max_process_interval"]),
        )
        incoming_qtimes[seq_to_idx[qspec.end_seq]].append(qspec)
        if (
            qspec.min_interval is not None
            and qspec.start_seq in seq_to_idx
            and qspec.end_seq in seq_to_idx
            and seq_to_idx[qspec.end_seq] == seq_to_idx[qspec.start_seq] + 1
            and qspec.start_type == "end"
            and qspec.end_type == "start"
        ):
            sequential_waits[seq_to_idx[qspec.start_seq]] = max(
                sequential_waits[seq_to_idx[qspec.start_seq]],
                qspec.min_interval,
            )

    for seq in process_order:
        proc_payload = path["process_list"][seq]
        candidates = tuple(
            sorted(
                (
                    CandidateSpec(
                        machine_id=machine_id,
                        process_time=to_int(info["process_time"]),
                        priority=to_float(info["priority"]),
                        batch_size=max(1, int(float(info.get("curr_batch_size", 1) or 1))),
                    )
                    for machine_id, info in proc_payload["eqp_list"].items()
                ),
                key=lambda item: (item.priority, item.process_time, item.machine_id),
            )
        )
        processes.append(
            ProcessSpec(
                seq=seq,
                proc_id=str((task_id, path_id, seq)),
                is_batch=bool(proc_payload["is_batch_type"]),
                batch_family=tuple(str(item) for item in proc_payload.get("batch_family", ())),
                diff_factory_info=tuple(tuple(pair) for pair in proc_payload["diff_factory_info"]),
                candidates=candidates,
                min_process_time=min(item.process_time for item in candidates),
            )
        )

    optimistic_total_from = [0 for _ in range(len(process_order) + 1)]
    optimistic_nonbatch_from = [0 for _ in range(len(process_order) + 1)]
    # 乐观尾部时间是候选动作评分使用的下界。
    # 该估计不提前扣除 setup、机器竞争和多数日历冲突，具体可行性由动作评估处理。
    for idx in range(len(process_order) - 1, -1, -1):
        proc = processes[idx]
        wait_after = sequential_waits[idx] if idx < len(process_order) - 1 else 0
        optimistic_total_from[idx] = (
            proc.min_process_time + wait_after + optimistic_total_from[idx + 1]
        )
        optimistic_nonbatch_from[idx] = optimistic_nonbatch_from[idx + 1]
        if not proc.is_batch:
            optimistic_nonbatch_from[idx] += proc.min_process_time

    return TaskSpec(
        task_id=task_id,
        earliest_ava_time=to_int(task_payload["earliest_ava_time"]),
        delivery_time=to_int(task_payload["task_delivery_time"]),
        priority=to_float(task_payload["task_priority"]),
        weight=to_float(task_payload["final_product_weight"]),
        path_id=path_id,
        process_order=process_order,
        seq_to_idx=seq_to_idx,
        processes=tuple(processes),
        incoming_qtimes=tuple(tuple(items) for items in incoming_qtimes),
        optimistic_total_from=tuple(optimistic_total_from),
        optimistic_nonbatch_from=tuple(optimistic_nonbatch_from),
    )


def build_instance(
    root: Path,
    input_path: Path,
    cache_path: Path,
    force: bool = False,
    path_nonbatch_mult: float = 3.0,
    path_batch_weight: float = 1.0,
    path_wait_weight: float = 1.0,
    path_machine_penalties: Optional[dict[str, float]] = None,
    force_path_map: Optional[dict[str, str]] = None,
    current_time_override: Optional[int] = None,
    zero_current_time: bool = False,
    maintenance_shift: Optional[int] = 0,
) -> InstanceData:
    """把大型 JSON 算例解析成可缓存的紧凑对象。"""
    strategy_key = path_strategy_signature(
        path_nonbatch_mult=path_nonbatch_mult,
        path_batch_weight=path_batch_weight,
        path_wait_weight=path_wait_weight,
        path_machine_penalties=path_machine_penalties or {},
        force_path_map=force_path_map or {},
        current_time_override=current_time_override,
        zero_current_time=zero_current_time,
        maintenance_shift=maintenance_shift,
    )
    if cache_path.exists() and not force and cache_path.stat().st_mtime >= input_path.stat().st_mtime:
        try:
            register_pickle_compat_aliases()
            with cache_path.open("rb") as fh:
                cached = pickle.load(fh)
            cached_path = getattr(cached, "source_input", None)
            cached_size = getattr(cached, "source_size", None)
            cached_mtime = getattr(cached, "source_mtime_ns", None)
            cached_strategy_key = getattr(cached, "path_strategy_key", "default")
            sig_path, sig_size, sig_mtime = input_signature(input_path)
            if (
                cached_path == sig_path
                and (cached_size is None or cached_size == sig_size)
                and (cached_mtime is None or cached_mtime == sig_mtime)
                and cached_strategy_key == strategy_key
            ):
                return cached
            print("[instance] cache mismatch detected, rebuilding", flush=True)
        except Exception as exc:
            print(f"[instance] cache reload failed, rebuilding: {exc}", flush=True)

    current_time: Optional[int] = None
    current_dt: Optional[str] = None
    horizon: Optional[int] = None
    with input_path.open("rb") as fh:
        current_time = next(ijson.items(fh, "time.current_time"))
    with input_path.open("rb") as fh:
        current_dt = next(ijson.items(fh, "time.current_date_time"))
    with input_path.open("rb") as fh:
        horizon = next(ijson.items(fh, "config.max_output_horizon"))

    raw_current_time = int(current_time)
    if zero_current_time:
        # 历史实验兼容模式：将输入 current_time 重新解释为分钟 0，
        # 并可按原 current_time 平移维修窗口。
        effective_current_time = 0
        effective_maintenance_shift = (
            raw_current_time if maintenance_shift is None else int(maintenance_shift)
        )
    else:
        effective_current_time = (
            raw_current_time if current_time_override is None else int(current_time_override)
        )
        effective_maintenance_shift = 0 if maintenance_shift is None else int(maintenance_shift)

    machines: dict[str, MachineSpec] = {}
    with input_path.open("rb") as fh:
        for machine_id, payload in ijson.kvitems(fh, "eqp"):
            intervals = []
            for start, end in payload["eqp_down_interval"]:
                # 内部统一把维修窗口存为半开区间 [start, end)。
                # 输入中的结束点按闭区间理解，因此转换为半开区间时需要加 1 分钟。
                intervals.append(
                    (
                        int(start) + effective_maintenance_shift,
                        int(end) + effective_maintenance_shift + 1,
                    )
                )
            intervals.sort()
            machines[machine_id] = MachineSpec(
                machine_id=machine_id,
                factory=str(payload["factory_info"]),
                down_intervals=tuple(intervals),
            )

    transitions: dict[str, dict[str, int]] = {}
    with input_path.open("rb") as fh:
        for from_machine, row in ijson.kvitems(fh, "transition"):
            transitions[from_machine] = {to_machine: int(value) for to_machine, value in row.items()}

    tasks: dict[str, TaskSpec] = {}
    count = 0
    with input_path.open("rb") as fh:
        for task_id, payload in ijson.kvitems(fh, "task"):
            tasks[task_id] = build_task_spec(
                task_id,
                payload,
                path_nonbatch_mult=path_nonbatch_mult,
                path_batch_weight=path_batch_weight,
                path_wait_weight=path_wait_weight,
                path_machine_penalties=path_machine_penalties,
                force_path_map=force_path_map,
            )
            count += 1
            if count % 250 == 0:
                print(f"[instance] parsed tasks: {count}", flush=True)

    instance = InstanceData(
        source_input=normalize_input_path(input_path),
        source_size=input_path.stat().st_size,
        source_mtime_ns=input_path.stat().st_mtime_ns,
        current_time=effective_current_time,
        current_dt=str(current_dt),
        start_dt=parse_dt(str(current_dt)),
        horizon=int(horizon),
        machines=machines,
        transitions=transitions,
        tasks=tasks,
        path_strategy_key=strategy_key,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(instance, fh, protocol=4)
    return instance


class RelaxedRLScheduler:
    """面向 FJSP 派工环境的两阶段确定性策略。

    第一阶段只接受乐观尾部仍可能在截止时间前完成的任务，以提高窗口内产量。
    第二阶段补齐所有剩余任务，保证输出是完整排产。有限组批标志会让
    p-batch 工序占用机器时间线，否则它们按 relaxed 无限产能处理。
    """

    def __init__(
        self,
        instance: InstanceData,
        setup_store: SetupRowStore,
        lookahead: int = 70,
        start_guard: int = 720,
        score_weight: float = 335.0,
        score_density: float = 23200.0,
        score_started: float = 140.0,
        score_family: float = 340.0,
        score_progress: float = 0.0,
        score_zero_setup: float = 380.0,
        score_setup_fixed: float = 430.0,
        score_setup_per: float = 4.4,
        score_est_final_per: float = 0.01,
        task_bonus_map: Optional[dict[str, float]] = None,
        defer_task_ids: Optional[set[str]] = None,
        force_machine_map: Optional[dict[tuple[str, str], str]] = None,
        phase2_started: float = 2200.0,
        phase2_density: float = 6800.0,
        phase2_family: float = 500.0,
        phase2_progress: float = 0.0,
        phase2_zero_setup: float = 900.0,
        phase2_setup_fixed: float = 540.0,
        phase2_setup_per: float = 4.4,
        phase2_finish_per: float = 0.01,
        phase2_started_gate: bool = True,
        finite_batch_capacity: bool = False,
        batch_group_wait: int = 0,
        batch_group_mixed_time: bool = False,
        batch_group_any_time: bool = False,
    ) -> None:
        self.instance = instance
        self.setup_store = setup_store
        self.lookahead = lookahead
        self.start_guard = start_guard
        self.score_weight = score_weight
        self.score_density = score_density
        self.score_started = score_started
        self.score_family = score_family
        self.score_progress = score_progress
        self.score_zero_setup = score_zero_setup
        self.score_setup_fixed = score_setup_fixed
        self.score_setup_per = score_setup_per
        self.score_est_final_per = score_est_final_per
        self.task_bonus_map = task_bonus_map or {}
        self.defer_task_ids = defer_task_ids or set()
        self.force_machine_map = force_machine_map or {}
        self.phase2_started = phase2_started
        self.phase2_density = phase2_density
        self.phase2_family = phase2_family
        self.phase2_progress = phase2_progress
        self.phase2_zero_setup = phase2_zero_setup
        self.phase2_setup_fixed = phase2_setup_fixed
        self.phase2_setup_per = phase2_setup_per
        self.phase2_finish_per = phase2_finish_per
        self.phase2_started_gate = phase2_started_gate
        self.finite_batch_capacity = finite_batch_capacity
        self.batch_group_wait = max(0, batch_group_wait)
        self.batch_group_mixed_time = batch_group_mixed_time
        self.batch_group_any_time = batch_group_any_time

        self.machine_free = {machine_id: instance.current_time for machine_id in instance.machines}
        self.machine_last_proc = {machine_id: None for machine_id in instance.machines}
        self.machine_last_family = {machine_id: None for machine_id in instance.machines}

        self.next_idx = {task_id: 0 for task_id in instance.tasks}
        self.task_records: dict[str, list[ScheduledOp]] = defaultdict(list)
        self.proc_records: dict[str, ScheduledOp] = {}
        self.task_status = {task_id: "active" for task_id in instance.tasks}
        self.setup_count = 0

    def fit_after_maintenance(self, machine_id: str, earliest_start: int, duration: int) -> int:
        """返回避开机器维修窗口后的最早可开工时间。"""
        start = max(earliest_start, self.instance.current_time)
        for down_start, down_end in self.instance.machines[machine_id].down_intervals:
            if start + duration <= down_start:
                return start
            if start >= down_end:
                continue
            start = down_end
        return start

    def transition_time(self, from_machine: str, to_machine: str) -> int:
        return self.instance.transitions.get(from_machine, {}).get(to_machine, 0)

    def transfer_allowed(self, prev_proc: ProcessSpec, from_machine: str, to_machine: str) -> bool:
        from_factory = self.instance.machines[from_machine].factory
        to_factory = self.instance.machines[to_machine].factory
        if from_factory == to_factory:
            return True
        return (from_factory, to_factory) in prev_proc.diff_factory_info

    def compute_bounds(self, task: TaskSpec, idx: int, machine_id: str, process_time: int) -> Optional[tuple[int, int]]:
        """计算一个动作的释放、前序、转运和 q-time 时间边界。"""
        proc = task.processes[idx]
        lower = max(self.instance.current_time, task.earliest_ava_time)
        upper = INF

        if idx > 0:
            prev_proc = task.processes[idx - 1]
            prev_record = self.proc_records[prev_proc.proc_id]
            if not self.transfer_allowed(prev_proc, prev_record.machine_id, machine_id):
                return None
            lower = max(
                lower,
                prev_record.finish + self.transition_time(prev_record.machine_id, machine_id),
            )

        for qtime in task.incoming_qtimes[idx]:
            start_record = self.proc_records[task.processes[task.seq_to_idx[qtime.start_seq]].proc_id]
            anchor = start_record.start if qtime.start_type == "start" else start_record.finish
            offset = process_time if qtime.end_type == "end" else 0
            if qtime.min_interval is not None:
                lower = max(lower, anchor + qtime.min_interval - offset)
            if qtime.max_interval is not None:
                upper = min(upper, anchor + qtime.max_interval - offset)

        if lower > upper:
            return None
        return lower, upper

    def has_forward_compatibility(self, task: TaskSpec, idx: int, machine_id: str) -> bool:
        """排除会导致下一道跨厂转运不可行的机器选择。"""
        if idx >= len(task.processes) - 1:
            return True
        current_proc = task.processes[idx]
        next_proc = task.processes[idx + 1]
        return any(
            self.transfer_allowed(current_proc, machine_id, candidate.machine_id)
            for candidate in next_proc.candidates
        )

    def process_family(self, proc: ProcessSpec) -> tuple[Any, ...]:
        """用于同工艺/零 setup 连续奖励的 family 键。"""
        return (
            proc.seq,
            proc.is_batch,
            tuple((candidate.machine_id, candidate.process_time) for candidate in proc.candidates),
        )

    def candidate_for_eval(self, eval_item: CandidateEval) -> CandidateSpec:
        task = self.instance.tasks[eval_item.task_id]
        proc = task.processes[eval_item.idx]
        candidate = next(
            item for item in proc.candidates if item.machine_id == eval_item.machine_id
        )
        return candidate

    def batch_family(self, proc: ProcessSpec) -> tuple[str, ...]:
        return getattr(proc, "batch_family", ())

    def evaluate_candidate(self, task_id: str, machine_id: str) -> Optional[CandidateEval]:
        """构造可行动作；若任一硬约束失败则返回 None。"""
        task = self.instance.tasks[task_id]
        idx = self.next_idx[task_id]
        proc = task.processes[idx]
        forced_machine = self.force_machine_map.get((task_id, proc.seq))
        if forced_machine is not None and machine_id != forced_machine:
            return None
        candidate = next((item for item in proc.candidates if item.machine_id == machine_id), None)
        if candidate is None:
            return None
        if not self.has_forward_compatibility(task, idx, machine_id):
            return None

        bounds = self.compute_bounds(task, idx, machine_id, candidate.process_time)
        if bounds is None:
            return None
        lower, upper = bounds
        setup_time = 0
        same_family = False
        zero_setup = False
        if not proc.is_batch or self.finite_batch_capacity:
            lower = max(lower, self.machine_free[machine_id])
            if not proc.is_batch:
                # 仅普通机器序列计入 setup。relaxed 轨道下，
                # 组批工序不占用机器时间线。
                setup_time = self.setup_store.get(self.machine_last_proc[machine_id], proc.proc_id)
                lower = max(lower, self.machine_free[machine_id] + setup_time)
                same_family = self.machine_last_family[machine_id] == self.process_family(proc)
                zero_setup = self.machine_last_proc[machine_id] is not None and setup_time == 0

        start = self.fit_after_maintenance(machine_id, lower, candidate.process_time)
        if start > upper:
            return None
        finish = start + candidate.process_time
        est_final = finish + task.optimistic_total_from[idx + 1]
        return CandidateEval(
            task_id=task_id,
            idx=idx,
            machine_id=machine_id,
            start=start,
            finish=finish,
            setup_time=setup_time,
            est_final=est_final,
            upper_bound=upper,
            option_priority=candidate.priority,
            started=bool(self.task_records[task_id]),
            same_family=same_family,
            zero_setup=zero_setup,
        )

    def batch_choice(self, task_id: str) -> Optional[CandidateEval]:
        """在普通派工前贪心推进 relaxed 组批前缀。"""
        task = self.instance.tasks[task_id]
        idx = self.next_idx[task_id]
        proc = task.processes[idx]
        best: Optional[tuple[float, CandidateEval]] = None
        for candidate in proc.candidates:
            eval_item = self.evaluate_candidate(task_id, candidate.machine_id)
            if eval_item is None:
                continue
            projected = eval_item.est_final
            score = -projected - candidate.priority * 10.0
            if best is None or score > best[0]:
                best = (score, eval_item)
        return None if best is None else best[1]

    def record_schedule(self, eval_item: CandidateEval) -> None:
        """把一个非合批工序写入调度状态。"""
        task = self.instance.tasks[eval_item.task_id]
        proc = task.processes[eval_item.idx]
        entry = ScheduledOp(
            task_id=task.task_id,
            seq=proc.seq,
            path_id=task.path_id,
            machine_id=eval_item.machine_id,
            start=eval_item.start,
            finish=eval_item.finish,
        )
        self.task_records[task.task_id].append(entry)
        self.proc_records[proc.proc_id] = entry
        self.next_idx[task.task_id] += 1

        if not proc.is_batch:
            if eval_item.setup_time > 0:
                self.setup_count += 1
            self.machine_free[eval_item.machine_id] = eval_item.finish
            self.machine_last_proc[eval_item.machine_id] = proc.proc_id
            self.machine_last_family[eval_item.machine_id] = self.process_family(proc)
        elif self.finite_batch_capacity:
            self.machine_free[eval_item.machine_id] = eval_item.finish

    def record_batch_group(
        self,
        best: CandidateEval,
        candidate_pool: list[CandidateEval],
    ) -> None:
        """在一台机器上写入一个有限 p-batch 批组。

        被选中的候选作为锚点。其他同 family 候选如果时间足够接近、
        容量允许，并且能共享同一个避开维修的开工时间和最大批时长，
        就可以加入该批。
        """
        task = self.instance.tasks[best.task_id]
        proc = task.processes[best.idx]
        chosen = self.candidate_for_eval(best)
        family = self.batch_family(proc)
        process_time = chosen.process_time
        capacity = getattr(chosen, "batch_size", 1)
        group = [best]
        group_start = best.start

        extras = [
            item
            for item in candidate_pool
            if item.task_id != best.task_id and item.machine_id == best.machine_id
        ]
        extras.sort(
            key=lambda item: (
                self.instance.tasks[item.task_id].weight,
                -item.option_priority,
                -item.start,
                item.task_id,
            ),
            reverse=True,
        )
        for item in extras:
            extra_task = self.instance.tasks[item.task_id]
            extra_proc = extra_task.processes[item.idx]
            if not extra_proc.is_batch:
                continue
            extra_candidate = self.candidate_for_eval(item)
            if self.batch_family(extra_proc) != family:
                continue
            if extra_candidate.process_time != process_time and (
                not (self.batch_group_mixed_time or self.batch_group_any_time)
                or (
                    self.batch_group_mixed_time
                    and not self.batch_group_any_time
                    and extra_candidate.process_time > process_time
                )
            ):
                continue
            if item.start > best.start + self.batch_group_wait:
                continue
            next_start = max(group_start, item.start)
            next_process_time = max(process_time, extra_candidate.process_time)
            if next_start > best.upper_bound or next_start > item.upper_bound:
                continue
            if any(next_start > member.upper_bound for member in group):
                continue
            if self.fit_after_maintenance(best.machine_id, next_start, next_process_time) != next_start:
                continue
            next_capacity = min(capacity, getattr(extra_candidate, "batch_size", 1))
            if len(group) + 1 > next_capacity:
                continue
            group.append(item)
            capacity = next_capacity
            group_start = next_start
            process_time = next_process_time

        finish = group_start + process_time
        for item in group:
            group_task = self.instance.tasks[item.task_id]
            group_proc = group_task.processes[item.idx]
            entry = ScheduledOp(
                task_id=group_task.task_id,
                seq=group_proc.seq,
                path_id=group_task.path_id,
                machine_id=best.machine_id,
                start=group_start,
                finish=finish,
            )
            self.task_records[group_task.task_id].append(entry)
            self.proc_records[group_proc.proc_id] = entry
            self.next_idx[group_task.task_id] += 1
            self.task_status[group_task.task_id] = "active"
        self.machine_free[best.machine_id] = finish

    def record_selected(
        self,
        best: CandidateEval,
        candidate_pool: list[CandidateEval],
    ) -> None:
        """根据轨道口径写入单工序或有限组批批组。"""
        task = self.instance.tasks[best.task_id]
        proc = task.processes[best.idx]
        if self.finite_batch_capacity and proc.is_batch:
            self.record_batch_group(best, candidate_pool)
        else:
            self.record_schedule(best)

    def advance_batch_prefix(self, task_id: str, respect_horizon: bool) -> None:
        """自动排入当前任务的 relaxed p-batch 前缀。

        无限产能轨道中，p-batch 工序不消耗共享机器时间线，
        因此只要任务内部约束允许，就可以立即向前推进。
        """
        if self.finite_batch_capacity:
            return
        while self.task_status[task_id] in {"active", "deferred"}:
            task = self.instance.tasks[task_id]
            idx = self.next_idx[task_id]
            if idx >= len(task.processes):
                self.task_status[task_id] = "done"
                return
            if not task.processes[idx].is_batch:
                return
            choice = self.batch_choice(task_id)
            if choice is None:
                self.task_status[task_id] = "infeasible"
                return
            if respect_horizon and choice.est_final > self.instance.horizon:
                self.task_status[task_id] = "deferred"
                return
            self.record_schedule(choice)

    def score_candidate(self, eval_item: CandidateEval, min_start: int) -> tuple[float, int, int, str]:
        """第一阶段评分：尽量提高截止产量，同时抑制 setup 失控。"""
        task = self.instance.tasks[eval_item.task_id]
        remaining_nb = max(task.optimistic_nonbatch_from[eval_item.idx], 1)
        density = task.weight / remaining_nb
        progress = eval_item.idx / max(len(task.processes) - 1, 1)
        q_slack = eval_item.upper_bound - eval_item.start
        q_bonus = 0.0 if q_slack >= INF // 2 else 5000.0 / (q_slack + 30.0)
        score = 0.0
        score += task.weight * self.score_weight
        score += density * self.score_density
        score += self.task_bonus_map.get(eval_item.task_id, 0.0)
        score += self.score_started if eval_item.started else 0.0
        score += self.score_family if eval_item.same_family else 0.0
        score += progress * self.score_progress
        score += self.score_zero_setup if eval_item.zero_setup else 0.0
        score += q_bonus
        score -= self.score_setup_fixed if eval_item.setup_time > 0 else 0.0
        score -= eval_item.setup_time * self.score_setup_per
        score -= (eval_item.start - min_start) * 0.05
        score -= eval_item.option_priority * 8.0
        score -= (eval_item.est_final - self.instance.current_time) * self.score_est_final_per
        return (score, -eval_item.start, -eval_item.finish, eval_item.task_id)

    def score_candidate_phase2(self, eval_item: CandidateEval, min_start: int) -> tuple[float, int, int, str]:
        """第二阶段评分：补齐剩余任务，同时偏好更紧凑的尾部排产。"""
        task = self.instance.tasks[eval_item.task_id]
        remaining_nb = max(task.optimistic_nonbatch_from[eval_item.idx], 1)
        progress = eval_item.idx / max(len(task.processes) - 1, 1)
        q_slack = eval_item.upper_bound - eval_item.start
        q_bonus = 0.0 if q_slack >= INF // 2 else 6000.0 / (q_slack + 30.0)
        score = 0.0
        score += self.phase2_started if eval_item.started else 0.0
        score += (task.weight / remaining_nb) * self.phase2_density
        score += self.phase2_family if eval_item.same_family else 0.0
        score += progress * self.phase2_progress
        score += self.phase2_zero_setup if eval_item.zero_setup else 0.0
        score += q_bonus
        score -= self.phase2_setup_fixed if eval_item.setup_time > 0 else 0.0
        score -= eval_item.setup_time * self.phase2_setup_per
        score -= (eval_item.start - min_start) * 0.08
        score -= (eval_item.finish - self.instance.current_time) * self.phase2_finish_per
        score -= eval_item.option_priority * 10.0
        return (score, -eval_item.start, -eval_item.finish, eval_item.task_id)

    def rebuild_state_with_kept_tasks(self, kept_task_ids: set[str]) -> None:
        """保留一组已完整任务，并重建调度状态。"""
        kept_records = {
            task_id: sorted(self.task_records[task_id], key=lambda item: int(item.seq))
            for task_id in kept_task_ids
        }

        self.machine_free = {machine_id: self.instance.current_time for machine_id in self.instance.machines}
        self.machine_last_proc = {machine_id: None for machine_id in self.instance.machines}
        self.machine_last_family = {machine_id: None for machine_id in self.instance.machines}
        self.proc_records = {}
        self.setup_count = 0

        new_task_records: dict[str, list[ScheduledOp]] = defaultdict(list)
        for task_id, records in kept_records.items():
            new_task_records[task_id].extend(records)
            task = self.instance.tasks[task_id]
            for idx, record in enumerate(records):
                self.proc_records[task.processes[idx].proc_id] = record

        machine_ops: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for task_id, records in kept_records.items():
            task = self.instance.tasks[task_id]
            for idx, record in enumerate(records):
                if not task.processes[idx].is_batch or self.finite_batch_capacity:
                    machine_ops[record.machine_id].append((record.start, record.finish, task.processes[idx].proc_id))

        proc_to_family = {}
        batch_proc_ids: set[str] = set()
        for task_id in kept_records:
            task = self.instance.tasks[task_id]
            for proc in task.processes:
                proc_to_family[proc.proc_id] = self.process_family(proc)
                if proc.is_batch:
                    batch_proc_ids.add(proc.proc_id)

        for machine_id, ops in machine_ops.items():
            ops.sort()
            prev_proc_id = None
            prev_finish = self.instance.current_time
            for _start, finish, proc_id in ops:
                proc_is_batch = proc_id in batch_proc_ids
                if (
                    not proc_is_batch
                    and prev_proc_id is not None
                    and self.setup_store.get(prev_proc_id, proc_id) > 0
                ):
                    self.setup_count += 1
                if not proc_is_batch:
                    prev_proc_id = proc_id
                prev_finish = finish
            self.machine_free[machine_id] = prev_finish
            self.machine_last_proc[machine_id] = prev_proc_id
            self.machine_last_family[machine_id] = proc_to_family.get(prev_proc_id)

        self.task_records = new_task_records
        for task_id, task in self.instance.tasks.items():
            prefix_len = len(kept_records.get(task_id, []))
            self.next_idx[task_id] = prefix_len
            if prefix_len >= len(task.processes):
                self.task_status[task_id] = "done"
            else:
                self.task_status[task_id] = "active"

    def repair_incomplete_tasks(self, max_rounds: int = 6) -> None:
        """迭代丢弃不完整任务并重跑第二阶段，直到尽量补全。"""
        previous_kept_count = -1
        for _ in range(max_rounds):
            kept_task_ids = {
                task_id
                for task_id, task in self.instance.tasks.items()
                if len(self.task_records.get(task_id, [])) == len(task.processes)
            }
            if len(kept_task_ids) == len(self.instance.tasks):
                return
            if len(kept_task_ids) == previous_kept_count:
                break
            previous_kept_count = len(kept_task_ids)

            self.rebuild_state_with_kept_tasks(kept_task_ids)
            self.run_phase2_completion_loop()

    def run_phase2_completion_loop(self) -> None:
        """求解和修复过程共用的第二阶段补齐循环。"""
        for task_id in self.instance.tasks:
            if self.task_status[task_id] == "active":
                self.advance_batch_prefix(task_id, respect_horizon=False)

        while True:
            feasible_candidates: list[CandidateEval] = []
            unfinished = 0
            for task_id, status in self.task_status.items():
                if status in {"done", "infeasible"}:
                    continue
                task = self.instance.tasks[task_id]
                idx = self.next_idx[task_id]
                if idx >= len(task.processes):
                    self.task_status[task_id] = "done"
                    continue
                unfinished += 1
                proc = task.processes[idx]
                if proc.is_batch and not self.finite_batch_capacity:
                    self.advance_batch_prefix(task_id, respect_horizon=False)
                    continue
                for candidate in proc.candidates:
                    eval_item = self.evaluate_candidate(task_id, candidate.machine_id)
                    if eval_item is not None:
                        feasible_candidates.append(eval_item)

            if not feasible_candidates:
                if unfinished == 0:
                    break
                for task_id, status in list(self.task_status.items()):
                    if status not in {"done", "infeasible"}:
                        self.task_status[task_id] = "infeasible"
                break

            started_candidates = [item for item in feasible_candidates if item.started]
            candidate_pool = (
                started_candidates
                if self.phase2_started_gate and started_candidates
                else feasible_candidates
            )
            # phase2 gate 保留“优先补齐已开工任务”的原始行为；
            # 只有实验显式允许时，未开工任务才会一起竞争。
            min_start = min(item.start for item in candidate_pool)
            shortlist = [
                item for item in candidate_pool if item.start <= min_start + max(self.lookahead, 360)
            ]
            best = max(shortlist, key=lambda item: self.score_candidate_phase2(item, min_start))
            self.record_selected(best, candidate_pool)
            self.task_status[best.task_id] = "active"
            self.advance_batch_prefix(best.task_id, respect_horizon=False)

    def find_internal_machine_violations(self) -> list[tuple[str, str, str, str, str]]:
        """检测修复重放状态时产生的 setup/重叠冲突。"""
        entries_by_machine: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
        for task_id, records in self.task_records.items():
            task = self.instance.tasks[task_id]
            sorted_records = sorted(records, key=lambda item: int(item.seq))
            for idx, record in enumerate(sorted_records):
                if idx >= len(task.processes):
                    break
                proc = task.processes[idx]
                if not proc.is_batch:
                    entries_by_machine[record.machine_id].append(
                        (record.start, record.finish, task_id, proc.proc_id)
                    )

        violations: list[tuple[str, str, str, str, str]] = []
        for machine_id, entries in entries_by_machine.items():
            entries.sort()
            prev_entry: Optional[tuple[int, int, str, str]] = None
            for entry in entries:
                if prev_entry is not None:
                    setup_time = self.setup_store.get(prev_entry[3], entry[3])
                    if entry[0] < prev_entry[1] + setup_time:
                        violations.append(
                            (machine_id, prev_entry[2], prev_entry[3], entry[2], entry[3])
                        )
                prev_entry = entry
        return violations

    def repair_setup_violations(self, max_rounds: int = 3) -> None:
        """围绕检测到的机器冲突丢弃相关任务，再重新补齐。"""
        for _ in range(max_rounds):
            violations = self.find_internal_machine_violations()
            if not violations:
                return
            reset_task_ids = {prev_task_id for _m, prev_task_id, _pp, _t, _cp in violations}
            reset_task_ids.update(task_id for _m, _pt, _pp, task_id, _cp in violations)
            kept_task_ids = {
                task_id
                for task_id, task in self.instance.tasks.items()
                if len(self.task_records.get(task_id, [])) == len(task.processes)
                and task_id not in reset_task_ids
            }
            if not kept_task_ids and reset_task_ids:
                # 若没有可保留任务，则维持当前状态，避免将排产整体清空。
                return
            self.rebuild_state_with_kept_tasks(kept_task_ids)
            self.run_phase2_completion_loop()

    def solve(self) -> dict[str, list[ScheduledOp]]:
        """执行第一阶段、第二阶段和轻量修复，生成完整排产。"""
        for task_id in self.instance.tasks:
            if task_id in self.defer_task_ids:
                self.task_status[task_id] = "deferred"
                continue
            self.advance_batch_prefix(task_id, respect_horizon=True)

        # 第一阶段：只派工乐观完工时间仍能落在 horizon 内的任务。
        # 其他任务推迟到补齐阶段，避免迟交尾部阻塞高价值准时任务。
        while True:
            feasible_candidates: list[CandidateEval] = []

            for task_id, status in self.task_status.items():
                if status != "active":
                    continue
                task = self.instance.tasks[task_id]
                idx = self.next_idx[task_id]
                if idx >= len(task.processes):
                    self.task_status[task_id] = "done"
                    continue
                proc = task.processes[idx]
                if proc.is_batch and not self.finite_batch_capacity:
                    self.advance_batch_prefix(task_id, respect_horizon=True)
                    continue

                best_finish = INF
                task_candidates: list[CandidateEval] = []
                for candidate in proc.candidates:
                    eval_item = self.evaluate_candidate(task_id, candidate.machine_id)
                    if eval_item is None:
                        continue
                    task_candidates.append(eval_item)
                    best_finish = min(best_finish, eval_item.est_final)
                if (
                    not self.task_records[task_id]
                    and best_finish > self.instance.horizon - self.start_guard
                ):
                    self.task_status[task_id] = "deferred"
                    continue
                if not task_candidates or best_finish > self.instance.horizon:
                    self.task_status[task_id] = "deferred"
                    continue
                feasible_candidates.extend(task_candidates)

            if not feasible_candidates:
                break

            min_start = min(item.start for item in feasible_candidates)
            shortlist = [
                item for item in feasible_candidates if item.start <= min_start + self.lookahead
            ]
            best = max(shortlist, key=lambda item: self.score_candidate(item, min_start))
            self.record_selected(best, feasible_candidates)
            self.task_status[best.task_id] = "active"
            self.advance_batch_prefix(best.task_id, respect_horizon=True)

        # 第二阶段：重新激活 deferred 任务，补完整个算例。
        for task_id, status in list(self.task_status.items()):
            if status == "deferred":
                self.task_status[task_id] = "active"
                self.advance_batch_prefix(task_id, respect_horizon=False)

        while True:
            feasible_candidates = []
            unfinished = 0

            for task_id, status in self.task_status.items():
                if status in {"done", "infeasible"}:
                    continue
                task = self.instance.tasks[task_id]
                idx = self.next_idx[task_id]
                if idx >= len(task.processes):
                    self.task_status[task_id] = "done"
                    continue
                unfinished += 1
                proc = task.processes[idx]
                if proc.is_batch and not self.finite_batch_capacity:
                    self.advance_batch_prefix(task_id, respect_horizon=False)
                    continue
                for candidate in proc.candidates:
                    eval_item = self.evaluate_candidate(task_id, candidate.machine_id)
                    if eval_item is not None:
                        feasible_candidates.append(eval_item)

            if not feasible_candidates:
                if unfinished == 0:
                    break
                for task_id, status in list(self.task_status.items()):
                    if status not in {"done", "infeasible"}:
                        self.task_status[task_id] = "infeasible"
                break

            min_start = min(item.start for item in feasible_candidates)
            started_candidates = [item for item in feasible_candidates if item.started]
            candidate_pool = (
                started_candidates
                if self.phase2_started_gate and started_candidates
                else feasible_candidates
            )
            # 第二阶段使用更宽候选窗口，降低长尾链路在修复阶段陷入局部死角的概率。
            min_start = min(item.start for item in candidate_pool)
            shortlist = [
                item for item in candidate_pool if item.start <= min_start + max(self.lookahead, 360)
            ]
            best = max(shortlist, key=lambda item: self.score_candidate_phase2(item, min_start))
            self.record_selected(best, candidate_pool)
            self.task_status[best.task_id] = "active"
            self.advance_batch_prefix(best.task_id, respect_horizon=False)

        self.repair_incomplete_tasks()
        self.repair_setup_violations()
        return self.task_records

    def metrics(self) -> dict[str, Any]:
        """计算生成后立即打印的求解器侧快速指标。"""
        completed_weight = 0.0
        completed_tasks = 0
        late_completed_tasks = 0
        scheduled_ops = sum(len(records) for records in self.task_records.values())
        scheduled_tasks = 0
        fully_scheduled_tasks = 0
        for task_id, records in self.task_records.items():
            if not records:
                continue
            scheduled_tasks += 1
            task = self.instance.tasks[task_id]
            if len(records) == len(task.processes):
                fully_scheduled_tasks += 1
                if records[-1].finish <= self.instance.horizon:
                    completed_tasks += 1
                    completed_weight += task.weight
                else:
                    late_completed_tasks += 1
        return {
            "scheduled_tasks": scheduled_tasks,
            "fully_scheduled_tasks": fully_scheduled_tasks,
            "total_tasks": len(self.instance.tasks),
            "scheduled_ops": scheduled_ops,
            "completed_tasks_within_horizon": completed_tasks,
            "completed_weight_within_horizon": round(completed_weight, 3),
            "late_completed_tasks": late_completed_tasks,
            "setup_count_positive": self.setup_count,
        }


def dump_solution(instance: InstanceData, task_records: dict[str, list[ScheduledOp]], output_path: Path) -> None:
    """把内部分钟轴排产结果写成提交/校验使用的 JSON 格式。

    内部记录只保存分钟值，输出文件要求日历时间字符串，因此本函数使用
    fmt_dt() 按 instance.current_time 和 instance.start_dt 做一次转换。
    """
    payload = {"task": {}}
    for task_id in sorted(task_records):
        records = sorted(task_records[task_id], key=lambda item: int(item.seq))
        if not records:
            continue
        payload["task"][task_id] = {"process_path": {}}
        for record in records:
            payload["task"][task_id]["process_path"][record.seq] = {
                "temp_machine_id": record.machine_id,
                "path_id": record.path_id,
                "process_start_time": fmt_dt(instance.start_dt, instance.current_time, record.start),
                "process_finish_time": fmt_dt(instance.start_dt, instance.current_time, record.finish),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_solution_records(output_path: Path) -> dict[str, list[ScheduledOp]]:
    """读取解文件，并把日历时间字符串转回 ScheduledOp 记录。

    注意：本函数读出的 start/finish 仍是 datetime。validate_solution()
    会再结合 instance.current_time 把它们转成分钟轴。
    """
    with output_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records: dict[str, list[ScheduledOp]] = {}
    for task_id, task_payload in payload.get("task", {}).items():
        items = []
        for seq, row in task_payload["process_path"].items():
            start = parse_dt(row["process_start_time"])
            finish = parse_dt(row["process_finish_time"])
            items.append(
                (
                    int(seq),
                    ScheduledOp(
                        task_id=task_id,
                        seq=str(seq),
                        path_id=str(row["path_id"]),
                        machine_id=str(row["temp_machine_id"]),
                        start=start,
                        finish=finish,
                    ),
                )
            )
        records[task_id] = [item for _, item in sorted(items)]
    return records


def validate_solution(
    instance: InstanceData,
    setup_store: SetupRowStore,
    output_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """按 relaxed 口径校验解文件并返回错误列表和指标。

    relaxed 口径的含义是：普通机器严格检查顺序、setup 和重叠；
    p-batch 组批工序不检查有限容量，也不要求同一批内 family/批时长一致。
    因此，有限组批解需要再用 validate_batch_solution.py 单独校验。

    校验内容包括：
    1. 输出是否包含所有任务，以及每个任务是否覆盖选中路径的全部工序。
    2. 每道工序的 path_id、候选机器、加工时长和维修窗口是否正确。
    3. 释放时间、前序约束、跨厂转运和 q-time 上下界是否满足。
    4. 普通机器上相邻工序是否留足 setup 时间，并统计正 setup 次数。
    5. 在 horizon 内完成的任务数量和产量。
    """
    raw_records = load_solution_records(output_path)
    errors: list[str] = []
    nonbatch_by_machine: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    completed_weight = 0.0
    completed_tasks = 0
    setup_count = 0
    fully_scheduled_tasks = 0

    missing_tasks = sorted(set(instance.tasks) - set(raw_records))
    if missing_tasks:
        errors.append(f"Missing tasks in output: {len(missing_tasks)}")

    for task_id, parsed in raw_records.items():
        if task_id not in instance.tasks:
            errors.append(f"Unknown task in output: {task_id}")
            continue
        task = instance.tasks[task_id]
        expected_prefix = list(task.process_order)
        actual_seqs = [record.seq for record in parsed]
        if actual_seqs != expected_prefix:
            errors.append(f"{task_id}: process seqs do not cover the full selected path")
            continue
        fully_scheduled_tasks += 1

        task_has_errors = False
        proc_records: dict[str, ScheduledOp] = {}
        for idx, record in enumerate(parsed):
            proc = task.processes[idx]
            if record.path_id != task.path_id:
                errors.append(f"{task_id}:{record.seq} path_id mismatch")
                task_has_errors = True
                continue

            machine_candidate = next(
                (item for item in proc.candidates if item.machine_id == record.machine_id),
                None,
            )
            if machine_candidate is None:
                errors.append(f"{task_id}:{record.seq} invalid machine {record.machine_id}")
                task_has_errors = True
                continue

            start_min = instance.current_time + int(
                (record.start - instance.start_dt).total_seconds() // 60
            )
            finish_min = instance.current_time + int(
                (record.finish - instance.start_dt).total_seconds() // 60
            )
            if finish_min - start_min != machine_candidate.process_time:
                errors.append(f"{task_id}:{record.seq} duration mismatch")
                task_has_errors = True
                continue

            for down_start, down_end in instance.machines[record.machine_id].down_intervals:
                if not (finish_min <= down_start or start_min >= down_end):
                    errors.append(f"{task_id}:{record.seq} overlaps maintenance")
                    task_has_errors = True
                    break

            scheduled = ScheduledOp(
                task_id=record.task_id,
                seq=record.seq,
                path_id=record.path_id,
                machine_id=record.machine_id,
                start=start_min,
                finish=finish_min,
            )
            proc_records[proc.proc_id] = scheduled
            if not proc.is_batch:
                nonbatch_by_machine[record.machine_id].append(
                    (start_min, finish_min, task_id, proc.proc_id)
                )

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
                    errors.append(f"{task_id}:{record.seq} missing predecessor record")
                    task_has_errors = True
                    continue
                from_factory = instance.machines[prev_record.machine_id].factory
                to_factory = instance.machines[proc_record.machine_id].factory
                if from_factory != to_factory and (
                    from_factory,
                    to_factory,
                ) not in prev_proc.diff_factory_info:
                    errors.append(f"{task_id}:{record.seq} illegal factory transfer")
                    task_has_errors = True
                lower = max(
                    lower,
                    prev_record.finish
                    + instance.transitions.get(prev_record.machine_id, {}).get(proc_record.machine_id, 0),
                )
            for qtime in task.incoming_qtimes[idx]:
                anchor_idx = task.seq_to_idx.get(qtime.start_seq)
                if anchor_idx is None:
                    errors.append(f"{task_id}:{record.seq} q-time anchor seq not in selected path")
                    task_has_errors = True
                    continue
                start_anchor = proc_records.get(task.processes[anchor_idx].proc_id)
                if start_anchor is None:
                    errors.append(f"{task_id}:{record.seq} missing q-time anchor record")
                    task_has_errors = True
                    continue
                anchor = start_anchor.start if qtime.start_type == "start" else start_anchor.finish
                offset = proc_record.finish - proc_record.start if qtime.end_type == "end" else 0
                if qtime.min_interval is not None:
                    lower = max(lower, anchor + qtime.min_interval - offset)
                if qtime.max_interval is not None:
                    upper = min(upper, anchor + qtime.max_interval - offset)
            if proc_record.start < lower:
                errors.append(f"{task_id}:{record.seq} violates lower time bound")
                task_has_errors = True
            if proc_record.start > upper:
                errors.append(f"{task_id}:{record.seq} violates upper time bound")
                task_has_errors = True

        if len(parsed) == len(task.processes):
            last = proc_records.get(task.processes[-1].proc_id)
            if last is not None and not task_has_errors and last.finish <= instance.horizon:
                completed_tasks += 1
                completed_weight += task.weight

    for machine_id, entries in nonbatch_by_machine.items():
        entries.sort()
        prev_finish = None
        prev_proc_id = None
        for start, finish, task_id, proc_id in entries:
            if prev_finish is not None:
                setup_time = setup_store.get(prev_proc_id, proc_id)
                if setup_time > 0:
                    setup_count += 1
                if start < prev_finish + setup_time:
                    errors.append(f"{machine_id}: overlap/setup violation between {prev_proc_id} and {proc_id}")
            prev_finish = finish
            prev_proc_id = proc_id

    metrics = {
        "completed_tasks_within_horizon": completed_tasks,
        "completed_weight_within_horizon": round(completed_weight, 3),
        "setup_count_positive": setup_count,
        "fully_scheduled_tasks": fully_scheduled_tasks,
        "total_tasks": len(instance.tasks),
        "machine_count_with_nonbatch_load": len(nonbatch_by_machine),
        "error_count": len(errors),
    }
    return errors, metrics


def parse_args() -> argparse.Namespace:
    """解析底层求解器命令行参数。

    该入口保留大量实验参数，适合复现调参过程、执行网格搜索或排查问题。
    若仅需使用当前预设参数求解，优先使用 solve_best_preset.py。
    """
    parser = argparse.ArgumentParser(description="RL-style relaxed FJSP solver")
    parser.add_argument("command", choices=["solve", "validate"])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input json path")
    parser.add_argument(
        "--horizon-override",
        type=int,
        default=None,
        help="Override config.max_output_horizon (minutes) for both scheduling and metrics",
    )
    parser.add_argument(
        "--current-time-override",
        type=int,
        default=None,
        help="Override time.current_time without editing the input JSON",
    )
    parser.add_argument(
        "--maintenance-shift",
        type=int,
        default=None,
        help="Shift all eqp_down_interval bounds by this many minutes",
    )
    parser.add_argument(
        "--zero-current-time",
        action="store_true",
        help=(
            "Treat the input current moment as minute 0. This sets current_time=0 "
            "and shifts maintenance windows by the original input current_time "
            "unless --maintenance-shift is also provided."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rl_relaxed_solution.json"),
        help="Output solution json",
    )
    parser.add_argument(
        "--instance-cache",
        type=Path,
        default=Path("cache/selected_instance.pkl"),
        help="Pickle cache for chosen paths and core data",
    )
    parser.add_argument(
        "--setup-db",
        type=Path,
        default=Path("cache/setup_rows.sqlite"),
        help="SQLite row-store for sparse setup matrix",
    )
    parser.add_argument("--path-nonbatch-mult", type=float, default=3.0, help="Path selection weight on non-batch processing time")
    parser.add_argument("--path-batch-weight", type=float, default=1.0, help="Path selection weight on batch processing time")
    parser.add_argument("--path-wait-weight", type=float, default=1.0, help="Path selection weight on minimum qtime waits")
    parser.add_argument(
        "--path-machine-penalty",
        type=str,
        default="",
        help="Comma-separated machine=penalty pairs applied during path selection",
    )
    parser.add_argument(
        "--force-path",
        type=str,
        default="",
        help="Comma-separated task_id=path_id overrides for path selection",
    )
    parser.add_argument(
        "--force-machine",
        type=str,
        default="",
        help="Comma-separated task_id:seq=machine_id overrides for operation machine selection",
    )
    parser.add_argument("--lookahead", type=int, default=70, help="Dispatch lookahead window in minutes")
    parser.add_argument("--start-guard", type=int, default=720, help="Defer starting tasks that cannot finish before horizon minus this slack")
    parser.add_argument("--score-weight", type=float, default=335.0)
    parser.add_argument("--score-density", type=float, default=23200.0)
    parser.add_argument("--score-started", type=float, default=140.0)
    parser.add_argument("--score-family", type=float, default=340.0)
    parser.add_argument("--score-progress", type=float, default=0.0)
    parser.add_argument("--score-zero-setup", type=float, default=380.0)
    parser.add_argument("--score-setup-fixed", type=float, default=430.0)
    parser.add_argument("--score-setup-per", type=float, default=4.4)
    parser.add_argument("--score-est-final-per", type=float, default=0.01)
    parser.add_argument(
        "--task-bonus",
        type=str,
        default="",
        help="Comma-separated task_id=score bonuses added to phase1 candidate scores",
    )
    parser.add_argument(
        "--defer-task",
        type=str,
        default="",
        help="Comma-separated task ids forced out of phase1 and completed only in phase2",
    )
    parser.add_argument("--phase2-started", type=float, default=2200.0)
    parser.add_argument("--phase2-density", type=float, default=6800.0)
    parser.add_argument("--phase2-family", type=float, default=500.0)
    parser.add_argument("--phase2-progress", type=float, default=0.0)
    parser.add_argument("--phase2-zero-setup", type=float, default=900.0)
    parser.add_argument("--phase2-setup-fixed", type=float, default=540.0)
    parser.add_argument("--phase2-setup-per", type=float, default=4.4)
    parser.add_argument("--phase2-finish-per", type=float, default=0.01)
    parser.add_argument(
        "--phase2-allow-unstarted",
        action="store_true",
        help="Let phase2 score all feasible tasks even when started-task candidates exist",
    )
    parser.add_argument(
        "--finite-batch-capacity",
        action="store_true",
        help="Treat batch machines as finite-capacity resources instead of relaxed infinite-capacity resources",
    )
    parser.add_argument(
        "--batch-group-wait",
        type=int,
        default=0,
        help="In finite-batch mode, let a selected batch wait this many minutes to form a same-family group",
    )
    parser.add_argument(
        "--batch-group-mixed-time",
        action="store_true",
        help="In finite-batch mode, allow same-family shorter batch tasks to join a longer selected batch",
    )
    parser.add_argument(
        "--batch-group-any-time",
        action="store_true",
        help="In finite-batch mode, allow same-family batch tasks with any process time and use the group maximum",
    )
    parser.add_argument("--rebuild-instance", action="store_true")
    parser.add_argument("--rebuild-setup", action="store_true")
    return parser.parse_args()


def main() -> int:
    """底层命令行入口：构建算例，执行求解或校验，并打印指标。

    command=solve 会先生成解文件，再立即调用 relaxed 校验逻辑；
    command=validate 只读取已有解文件并校验。自动化搜索脚本和预设入口
    均会复用本入口暴露的参数能力。
    """
    args = parse_args()
    root = args.root.resolve()
    input_path = args.input.resolve() if args.input else detect_input_json(root)
    output_path = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    instance_cache = (
        (root / args.instance_cache).resolve()
        if not args.instance_cache.is_absolute()
        else args.instance_cache
    )
    setup_db = (
        (root / args.setup_db).resolve() if not args.setup_db.is_absolute() else args.setup_db
    )
    path_machine_penalties = parse_named_float_map(args.path_machine_penalty)
    force_path_map = parse_named_str_map(args.force_path)
    force_machine_map = parse_force_machine_map(args.force_machine)
    task_bonus_map = parse_named_float_map(args.task_bonus)
    defer_task_ids = parse_name_set(args.defer_task)
    inferred_force_paths = False
    if args.command == "validate" and not force_path_map:
        force_path_map = infer_force_path_map_from_solution(output_path)
        inferred_force_paths = bool(force_path_map)

    print(f"[main] input: {input_path}", flush=True)
    if inferred_force_paths:
        print(
            f"[main] inferred force paths from output: {len(force_path_map)}",
            flush=True,
        )
    maintenance_shift = args.maintenance_shift
    if args.zero_current_time and maintenance_shift is None:
        with input_path.open("rb") as fh:
            maintenance_shift = int(next(ijson.items(fh, "time.current_time")))
    elif maintenance_shift is None:
        maintenance_shift = 0

    if path_machine_penalties or force_path_map or force_machine_map or task_bonus_map or defer_task_ids or any(
        value != default
        for value, default in (
            (args.path_nonbatch_mult, 3.0),
            (args.path_batch_weight, 1.0),
            (args.path_wait_weight, 1.0),
        )
    ) or args.current_time_override is not None or maintenance_shift != 0 or args.zero_current_time:
        print(
            json.dumps(
                {
                    "path_nonbatch_mult": args.path_nonbatch_mult,
                    "path_batch_weight": args.path_batch_weight,
                    "path_wait_weight": args.path_wait_weight,
                    "path_machine_penalty": path_machine_penalties,
                    "force_path": "<inferred from output>" if inferred_force_paths else force_path_map,
                    "force_machine": {
                        f"{task_id}:{seq}": machine_id
                        for (task_id, seq), machine_id in sorted(force_machine_map.items())
                    },
                    "task_bonus": task_bonus_map,
                    "defer_task": sorted(defer_task_ids),
                    "current_time_override": args.current_time_override,
                    "maintenance_shift": maintenance_shift,
                    "zero_current_time": args.zero_current_time,
                    "finite_batch_capacity": args.finite_batch_capacity,
                    "batch_group_wait": args.batch_group_wait,
                    "batch_group_mixed_time": args.batch_group_mixed_time,
                    "batch_group_any_time": args.batch_group_any_time,
                },
                ensure_ascii=False,
            ),
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
        print(f"[main] horizon override: {instance.horizon}", flush=True)
    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=args.rebuild_setup)

    try:
        if args.command == "solve":
            scheduler = RelaxedRLScheduler(
                instance,
                setup_store,
                lookahead=args.lookahead,
                start_guard=args.start_guard,
                score_weight=args.score_weight,
                score_density=args.score_density,
                score_started=args.score_started,
                score_family=args.score_family,
                score_progress=args.score_progress,
                score_zero_setup=args.score_zero_setup,
                score_setup_fixed=args.score_setup_fixed,
                score_setup_per=args.score_setup_per,
                score_est_final_per=args.score_est_final_per,
                task_bonus_map=task_bonus_map,
                defer_task_ids=defer_task_ids,
                force_machine_map=force_machine_map,
                phase2_started=args.phase2_started,
                phase2_density=args.phase2_density,
                phase2_family=args.phase2_family,
                phase2_progress=args.phase2_progress,
                phase2_zero_setup=args.phase2_zero_setup,
                phase2_setup_fixed=args.phase2_setup_fixed,
                phase2_setup_per=args.phase2_setup_per,
                phase2_finish_per=args.phase2_finish_per,
                phase2_started_gate=not args.phase2_allow_unstarted,
                finite_batch_capacity=args.finite_batch_capacity,
                batch_group_wait=args.batch_group_wait,
                batch_group_mixed_time=args.batch_group_mixed_time,
                batch_group_any_time=args.batch_group_any_time,
            )
            task_records = scheduler.solve()
            dump_solution(instance, task_records, output_path)
            metrics = scheduler.metrics()
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
            errors, validation = validate_solution(instance, setup_store, output_path)
            print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
            if errors:
                print("[validate] sample errors:", flush=True)
                for item in errors[:20]:
                    print(item, flush=True)
            else:
                print("[validate] no errors", flush=True)
        else:
            errors, validation = validate_solution(instance, setup_store, output_path)
            print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
            if errors:
                for item in errors[:50]:
                    print(item, flush=True)
                return 1
            print("[validate] no errors", flush=True)
        return 0
    finally:
        setup_store.close()


if __name__ == "__main__":
    sys.exit(main())

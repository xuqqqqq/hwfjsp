#!/usr/bin/env python3
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
    if value is None:
        return None
    return int(value)


def to_float(value: Any) -> float:
    return float(value)


def parse_dt(value: str) -> datetime:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime: {value}")


def fmt_dt(start_dt: datetime, current_time: int, value: int) -> str:
    return (start_dt + timedelta(minutes=value - current_time)).strftime("%Y/%m/%d %H:%M:%S")


def normalize_input_path(path: Path) -> str:
    return str(path.resolve())


def input_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (normalize_input_path(path), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class CandidateSpec:
    machine_id: str
    process_time: int
    priority: float


@dataclass(frozen=True)
class QTimeSpec:
    start_seq: str
    start_type: str
    end_seq: str
    end_type: str
    min_interval: Optional[int]
    max_interval: Optional[int]


@dataclass
class ProcessSpec:
    seq: str
    proc_id: str
    is_batch: bool
    diff_factory_info: tuple[tuple[str, str], ...]
    candidates: tuple[CandidateSpec, ...]
    min_process_time: int


@dataclass
class TaskSpec:
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
    machine_id: str
    factory: str
    down_intervals: tuple[tuple[int, int], ...]


@dataclass
class ScheduledOp:
    task_id: str
    seq: str
    path_id: str
    machine_id: str
    start: int
    finish: int


@dataclass
class CandidateEval:
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


class SetupRowStore:
    def __init__(self, db_path: Path, row_cache_size: int = 256) -> None:
        self.db_path = db_path
        self.row_cache_size = row_cache_size
        self.conn: Optional[sqlite3.Connection] = None
        self.row_cache: OrderedDict[str, dict[str, int]] = OrderedDict()

    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _remove_db_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(self.db_path) + suffix)
            if target.exists():
                target.unlink()

    def _db_matches_input(self, input_path: Path) -> bool:
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
        if not from_proc:
            return 0
        return self._load_row(from_proc).get(to_proc, 0)


def detect_input_json(root: Path) -> Path:
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


def choose_path(task_id: str, task_payload: dict[str, Any]) -> str:
    best_key: Optional[tuple[float, float, str]] = None
    best_path_id = ""
    for path_id, path in task_payload["process_path"].items():
        non_batch = 0.0
        batch = 0.0
        min_wait = 0.0
        for proc in path["process_list"].values():
            min_pt = min(to_int(info["process_time"]) for info in proc["eqp_list"].values())
            if proc["is_batch_type"]:
                batch += min_pt
            else:
                non_batch += min_pt
        for qtime in path.get("qtime_info", {}).values():
            if qtime["min_process_interval"] is not None:
                min_wait += float(qtime["min_process_interval"])
        key = (non_batch * 3.0 + batch + min_wait, non_batch + batch + min_wait, path_id)
        if best_key is None or key < best_key:
            best_key = key
            best_path_id = path_id
    if not best_path_id:
        raise ValueError(f"No path found for task {task_id}")
    return best_path_id


def build_task_spec(task_id: str, task_payload: dict[str, Any]) -> TaskSpec:
    path_id = choose_path(task_id, task_payload)
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
                diff_factory_info=tuple(tuple(pair) for pair in proc_payload["diff_factory_info"]),
                candidates=candidates,
                min_process_time=min(item.process_time for item in candidates),
            )
        )

    optimistic_total_from = [0 for _ in range(len(process_order) + 1)]
    optimistic_nonbatch_from = [0 for _ in range(len(process_order) + 1)]
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


def build_instance(root: Path, input_path: Path, cache_path: Path, force: bool = False) -> InstanceData:
    if cache_path.exists() and not force and cache_path.stat().st_mtime >= input_path.stat().st_mtime:
        try:
            with cache_path.open("rb") as fh:
                cached = pickle.load(fh)
            cached_path = getattr(cached, "source_input", None)
            cached_size = getattr(cached, "source_size", None)
            cached_mtime = getattr(cached, "source_mtime_ns", None)
            sig_path, sig_size, sig_mtime = input_signature(input_path)
            if (
                cached_path == sig_path
                and (cached_size is None or cached_size == sig_size)
                and (cached_mtime is None or cached_mtime == sig_mtime)
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

    machines: dict[str, MachineSpec] = {}
    with input_path.open("rb") as fh:
        for machine_id, payload in ijson.kvitems(fh, "eqp"):
            intervals = []
            for start, end in payload["eqp_down_interval"]:
                intervals.append((int(start), int(end) + 1))
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
            tasks[task_id] = build_task_spec(task_id, payload)
            count += 1
            if count % 250 == 0:
                print(f"[instance] parsed tasks: {count}", flush=True)

    instance = InstanceData(
        source_input=normalize_input_path(input_path),
        source_size=input_path.stat().st_size,
        source_mtime_ns=input_path.stat().st_mtime_ns,
        current_time=int(current_time),
        current_dt=str(current_dt),
        start_dt=parse_dt(str(current_dt)),
        horizon=int(horizon),
        machines=machines,
        transitions=transitions,
        tasks=tasks,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(instance, fh, protocol=4)
    return instance


class RelaxedRLScheduler:
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
        phase2_started: float = 2200.0,
        phase2_density: float = 6800.0,
        phase2_family: float = 500.0,
        phase2_progress: float = 0.0,
        phase2_zero_setup: float = 900.0,
        phase2_setup_fixed: float = 540.0,
        phase2_setup_per: float = 4.4,
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
        self.phase2_started = phase2_started
        self.phase2_density = phase2_density
        self.phase2_family = phase2_family
        self.phase2_progress = phase2_progress
        self.phase2_zero_setup = phase2_zero_setup
        self.phase2_setup_fixed = phase2_setup_fixed
        self.phase2_setup_per = phase2_setup_per

        self.machine_free = {machine_id: instance.current_time for machine_id in instance.machines}
        self.machine_last_proc = {machine_id: None for machine_id in instance.machines}
        self.machine_last_family = {machine_id: None for machine_id in instance.machines}

        self.next_idx = {task_id: 0 for task_id in instance.tasks}
        self.task_records: dict[str, list[ScheduledOp]] = defaultdict(list)
        self.proc_records: dict[str, ScheduledOp] = {}
        self.task_status = {task_id: "active" for task_id in instance.tasks}
        self.setup_count = 0

    def fit_after_maintenance(self, machine_id: str, earliest_start: int, duration: int) -> int:
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
        if idx >= len(task.processes) - 1:
            return True
        current_proc = task.processes[idx]
        next_proc = task.processes[idx + 1]
        return any(
            self.transfer_allowed(current_proc, machine_id, candidate.machine_id)
            for candidate in next_proc.candidates
        )

    def process_family(self, proc: ProcessSpec) -> tuple[Any, ...]:
        return (
            proc.seq,
            proc.is_batch,
            tuple((candidate.machine_id, candidate.process_time) for candidate in proc.candidates),
        )

    def evaluate_candidate(self, task_id: str, machine_id: str) -> Optional[CandidateEval]:
        task = self.instance.tasks[task_id]
        idx = self.next_idx[task_id]
        proc = task.processes[idx]
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
        if not proc.is_batch:
            lower = max(lower, self.machine_free[machine_id])
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

    def advance_batch_prefix(self, task_id: str, respect_horizon: bool) -> None:
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
        task = self.instance.tasks[eval_item.task_id]
        remaining_nb = max(task.optimistic_nonbatch_from[eval_item.idx], 1)
        density = task.weight / remaining_nb
        progress = eval_item.idx / max(len(task.processes) - 1, 1)
        q_slack = eval_item.upper_bound - eval_item.start
        q_bonus = 0.0 if q_slack >= INF // 2 else 5000.0 / (q_slack + 30.0)
        score = 0.0
        score += task.weight * self.score_weight
        score += density * self.score_density
        score += self.score_started if eval_item.started else 0.0
        score += self.score_family if eval_item.same_family else 0.0
        score += progress * self.score_progress
        score += self.score_zero_setup if eval_item.zero_setup else 0.0
        score += q_bonus
        score -= self.score_setup_fixed if eval_item.setup_time > 0 else 0.0
        score -= eval_item.setup_time * self.score_setup_per
        score -= (eval_item.start - min_start) * 0.05
        score -= eval_item.option_priority * 8.0
        score -= (eval_item.est_final - self.instance.current_time) * 0.01
        return (score, -eval_item.start, -eval_item.finish, eval_item.task_id)

    def score_candidate_phase2(self, eval_item: CandidateEval, min_start: int) -> tuple[float, int, int, str]:
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
        score -= (eval_item.finish - self.instance.current_time) * 0.01
        score -= eval_item.option_priority * 10.0
        return (score, -eval_item.start, -eval_item.finish, eval_item.task_id)

    def rebuild_state_with_kept_tasks(self, kept_task_ids: set[str]) -> None:
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
                if not task.processes[idx].is_batch:
                    machine_ops[record.machine_id].append((record.start, record.finish, task.processes[idx].proc_id))

        proc_to_family = {}
        for task_id in kept_task_ids:
            task = self.instance.tasks[task_id]
            for proc in task.processes:
                proc_to_family[proc.proc_id] = self.process_family(proc)

        for machine_id, ops in machine_ops.items():
            ops.sort()
            prev_proc_id = None
            prev_finish = self.instance.current_time
            for _start, finish, proc_id in ops:
                if prev_proc_id is not None and self.setup_store.get(prev_proc_id, proc_id) > 0:
                    self.setup_count += 1
                prev_proc_id = proc_id
                prev_finish = finish
            self.machine_free[machine_id] = prev_finish
            self.machine_last_proc[machine_id] = prev_proc_id
            self.machine_last_family[machine_id] = proc_to_family.get(prev_proc_id)

        self.task_records = new_task_records
        for task_id, task in self.instance.tasks.items():
            if task_id in kept_task_ids:
                self.next_idx[task_id] = len(task.processes)
                self.task_status[task_id] = "done"
            else:
                self.next_idx[task_id] = 0
                self.task_status[task_id] = "active"

    def repair_incomplete_tasks(self) -> None:
        kept_task_ids = {
            task_id
            for task_id, task in self.instance.tasks.items()
            if len(self.task_records.get(task_id, [])) == len(task.processes)
        }
        if len(kept_task_ids) == len(self.instance.tasks):
            return

        self.rebuild_state_with_kept_tasks(kept_task_ids)

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
                if proc.is_batch:
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
            candidate_pool = started_candidates if started_candidates else feasible_candidates
            min_start = min(item.start for item in candidate_pool)
            shortlist = [
                item for item in candidate_pool if item.start <= min_start + max(self.lookahead, 360)
            ]
            best = max(shortlist, key=lambda item: self.score_candidate_phase2(item, min_start))
            self.record_schedule(best)
            self.task_status[best.task_id] = "active"
            self.advance_batch_prefix(best.task_id, respect_horizon=False)

    def solve(self) -> dict[str, list[ScheduledOp]]:
        for task_id in self.instance.tasks:
            self.advance_batch_prefix(task_id, respect_horizon=True)

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
                if proc.is_batch:
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
            self.record_schedule(best)
            self.task_status[best.task_id] = "active"
            self.advance_batch_prefix(best.task_id, respect_horizon=True)

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
                if proc.is_batch:
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
            candidate_pool = started_candidates if started_candidates else feasible_candidates
            min_start = min(item.start for item in candidate_pool)
            shortlist = [
                item for item in candidate_pool if item.start <= min_start + max(self.lookahead, 360)
            ]
            best = max(shortlist, key=lambda item: self.score_candidate_phase2(item, min_start))
            self.record_schedule(best)
            self.task_status[best.task_id] = "active"
            self.advance_batch_prefix(best.task_id, respect_horizon=False)

        self.repair_incomplete_tasks()
        return self.task_records

    def metrics(self) -> dict[str, Any]:
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

        proc_records: dict[str, ScheduledOp] = {}
        for idx, record in enumerate(parsed):
            proc = task.processes[idx]
            if record.path_id != task.path_id:
                errors.append(f"{task_id}:{record.seq} path_id mismatch")
                continue

            machine_candidate = next(
                (item for item in proc.candidates if item.machine_id == record.machine_id),
                None,
            )
            if machine_candidate is None:
                errors.append(f"{task_id}:{record.seq} invalid machine {record.machine_id}")
                continue

            start_min = instance.current_time + int(
                (record.start - instance.start_dt).total_seconds() // 60
            )
            finish_min = instance.current_time + int(
                (record.finish - instance.start_dt).total_seconds() // 60
            )
            if finish_min - start_min != machine_candidate.process_time:
                errors.append(f"{task_id}:{record.seq} duration mismatch")
                continue

            for down_start, down_end in instance.machines[record.machine_id].down_intervals:
                if not (finish_min <= down_start or start_min >= down_end):
                    errors.append(f"{task_id}:{record.seq} overlaps maintenance")
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
                    continue
                from_factory = instance.machines[prev_record.machine_id].factory
                to_factory = instance.machines[proc_record.machine_id].factory
                if from_factory != to_factory and (
                    from_factory,
                    to_factory,
                ) not in prev_proc.diff_factory_info:
                    errors.append(f"{task_id}:{record.seq} illegal factory transfer")
                lower = max(
                    lower,
                    prev_record.finish
                    + instance.transitions.get(prev_record.machine_id, {}).get(proc_record.machine_id, 0),
                )
            for qtime in task.incoming_qtimes[idx]:
                start_anchor = proc_records[
                    task.processes[task.seq_to_idx[qtime.start_seq]].proc_id
                ]
                anchor = start_anchor.start if qtime.start_type == "start" else start_anchor.finish
                offset = proc_record.finish - proc_record.start if qtime.end_type == "end" else 0
                if qtime.min_interval is not None:
                    lower = max(lower, anchor + qtime.min_interval - offset)
                if qtime.max_interval is not None:
                    upper = min(upper, anchor + qtime.max_interval - offset)
            if proc_record.start < lower:
                errors.append(f"{task_id}:{record.seq} violates lower time bound")
            if proc_record.start > upper:
                errors.append(f"{task_id}:{record.seq} violates upper time bound")

        if len(parsed) == len(task.processes):
            last = proc_records[task.processes[-1].proc_id]
            if last.finish <= instance.horizon:
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
    parser.add_argument("--phase2-started", type=float, default=2200.0)
    parser.add_argument("--phase2-density", type=float, default=6800.0)
    parser.add_argument("--phase2-family", type=float, default=500.0)
    parser.add_argument("--phase2-progress", type=float, default=0.0)
    parser.add_argument("--phase2-zero-setup", type=float, default=900.0)
    parser.add_argument("--phase2-setup-fixed", type=float, default=540.0)
    parser.add_argument("--phase2-setup-per", type=float, default=4.4)
    parser.add_argument("--rebuild-instance", action="store_true")
    parser.add_argument("--rebuild-setup", action="store_true")
    return parser.parse_args()


def main() -> int:
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

    print(f"[main] input: {input_path}", flush=True)
    instance = build_instance(root, input_path, instance_cache, force=args.rebuild_instance)
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
                phase2_started=args.phase2_started,
                phase2_density=args.phase2_density,
                phase2_family=args.phase2_family,
                phase2_progress=args.phase2_progress,
                phase2_zero_setup=args.phase2_zero_setup,
                phase2_setup_fixed=args.phase2_setup_fixed,
                phase2_setup_per=args.phase2_setup_per,
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

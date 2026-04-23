#!/usr/bin/env python3
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rl_relaxed_solver import CandidateEval, InstanceData, RelaxedRLScheduler, SetupRowStore


@dataclass
class DispatchObservation:
    global_features: np.ndarray
    candidate_features: np.ndarray
    action_mask: np.ndarray
    heuristic_index: int
    candidate_keys: list[str]


def subset_instance(instance: InstanceData, task_ids: list[str]) -> InstanceData:
    task_map = {task_id: instance.tasks[task_id] for task_id in task_ids}
    return InstanceData(
        source_input=instance.source_input,
        source_size=instance.source_size,
        source_mtime_ns=instance.source_mtime_ns,
        current_time=instance.current_time,
        current_dt=instance.current_dt,
        start_dt=instance.start_dt,
        horizon=instance.horizon,
        machines=instance.machines,
        transitions=instance.transitions,
        tasks=task_map,
    )


def sample_task_ids(
    instance: InstanceData,
    task_limit: int,
    strategy: str,
    rng: random.Random,
) -> list[str]:
    all_ids = list(instance.tasks)
    if task_limit <= 0 or task_limit >= len(all_ids):
        return all_ids

    if strategy == "weight":
        ranked = sorted(
            all_ids,
            key=lambda task_id: (-instance.tasks[task_id].weight, instance.tasks[task_id].earliest_ava_time, task_id),
        )
        return ranked[:task_limit]
    if strategy == "earliest":
        ranked = sorted(
            all_ids,
            key=lambda task_id: (instance.tasks[task_id].earliest_ava_time, -instance.tasks[task_id].weight, task_id),
        )
        return ranked[:task_limit]
    if strategy == "mixed":
        by_weight = sorted(
            all_ids,
            key=lambda task_id: (-instance.tasks[task_id].weight, task_id),
        )
        by_early = sorted(
            all_ids,
            key=lambda task_id: (instance.tasks[task_id].earliest_ava_time, -instance.tasks[task_id].weight, task_id),
        )
        keep: list[str] = []
        seen: set[str] = set()
        half = max(task_limit // 2, 1)
        for task_id in by_weight[:half] + by_early[:task_limit]:
            if task_id in seen:
                continue
            keep.append(task_id)
            seen.add(task_id)
            if len(keep) >= task_limit:
                return keep
        if len(keep) < task_limit:
            remaining = [task_id for task_id in all_ids if task_id not in seen]
            rng.shuffle(remaining)
            keep.extend(remaining[: task_limit - len(keep)])
        return keep

    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    return shuffled[:task_limit]


class Phase1DispatchScheduler(RelaxedRLScheduler):
    def __init__(
        self,
        instance: InstanceData,
        setup_store: SetupRowStore,
        lookahead: int = 70,
        start_guard: int = 360,
        max_candidates: int = 32,
        reward_weight_scale: float = 100.0,
        reward_setup_penalty: float = 6.0,
        reward_delay_penalty: float = 0.2,
        reward_late_penalty: float = 12.0,
        reward_same_family_bonus: float = 0.5,
        reward_zero_setup_bonus: float = 0.75,
    ) -> None:
        super().__init__(
            instance,
            setup_store,
            lookahead=lookahead,
            start_guard=start_guard,
        )
        self.max_candidates = max_candidates
        self.reward_weight_scale = reward_weight_scale
        self.reward_setup_penalty = reward_setup_penalty
        self.reward_delay_penalty = reward_delay_penalty
        self.reward_late_penalty = reward_late_penalty
        self.reward_same_family_bonus = reward_same_family_bonus
        self.reward_zero_setup_bonus = reward_zero_setup_bonus
        self.rewarded_tasks: set[str] = set()
        self.completed_weight = 0.0
        self.completed_tasks = 0
        self.total_weight = sum(task.weight for task in instance.tasks.values()) or 1.0

    def bootstrap(self) -> None:
        for task_id in self.instance.tasks:
            self.advance_batch_prefix(task_id, respect_horizon=True)
        self._harvest_completed_weight()

    def _harvest_completed_weight(self) -> tuple[float, int]:
        gained_weight = 0.0
        gained_tasks = 0
        for task_id, task in self.instance.tasks.items():
            if task_id in self.rewarded_tasks:
                continue
            records = self.task_records.get(task_id, [])
            if len(records) != len(task.processes):
                continue
            if records[-1].finish > self.instance.horizon:
                continue
            self.rewarded_tasks.add(task_id)
            self.completed_weight += task.weight
            self.completed_tasks += 1
            gained_weight += task.weight
            gained_tasks += 1
        return gained_weight, gained_tasks

    def _active_status_counts(self) -> tuple[int, int, int]:
        active = sum(1 for status in self.task_status.values() if status == "active")
        deferred = sum(1 for status in self.task_status.values() if status == "deferred")
        done = sum(1 for status in self.task_status.values() if status == "done")
        return active, deferred, done

    def collect_phase1_candidates(self) -> list[CandidateEval]:
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

            best_finish = 10**18
            task_candidates: list[CandidateEval] = []
            for candidate in proc.candidates:
                eval_item = self.evaluate_candidate(task_id, candidate.machine_id)
                if eval_item is None:
                    continue
                task_candidates.append(eval_item)
                best_finish = min(best_finish, eval_item.est_final)
            if not self.task_records[task_id] and best_finish > self.instance.horizon - self.start_guard:
                self.task_status[task_id] = "deferred"
                continue
            if not task_candidates or best_finish > self.instance.horizon:
                self.task_status[task_id] = "deferred"
                continue
            feasible_candidates.extend(task_candidates)

        return feasible_candidates

    def build_observation(self, feasible_candidates: list[CandidateEval]) -> tuple[DispatchObservation, list[CandidateEval], int]:
        if not feasible_candidates:
            raise ValueError("Cannot build observation without candidates")

        min_start = min(item.start for item in feasible_candidates)
        shortlist = [item for item in feasible_candidates if item.start <= min_start + self.lookahead]
        ranked = sorted(
            shortlist,
            key=lambda item: self.score_candidate(item, min_start),
            reverse=True,
        )
        candidates = ranked[: self.max_candidates]
        if not candidates:
            candidates = ranked[:1]

        active_count, deferred_count, done_count = self._active_status_counts()
        horizon_span = max(self.instance.horizon - self.instance.current_time, 1)
        current_frontier = min_start
        global_features = np.asarray(
            [
                (current_frontier - self.instance.current_time) / horizon_span,
                (self.instance.horizon - current_frontier) / horizon_span,
                self.completed_weight / self.total_weight,
                self.completed_tasks / max(len(self.instance.tasks), 1),
                self.setup_count / max(len(self.instance.tasks), 1),
                active_count / max(len(self.instance.tasks), 1),
                deferred_count / max(len(self.instance.tasks), 1),
                done_count / max(len(self.instance.tasks), 1),
            ],
            dtype=np.float32,
        )

        candidate_features = np.zeros((self.max_candidates, 16), dtype=np.float32)
        action_mask = np.zeros((self.max_candidates,), dtype=np.float32)
        heuristic_index = 0
        candidate_keys: list[str] = []
        score_keys = [self.score_candidate(item, min_start) for item in candidates]
        best_key = max(score_keys)
        heuristic_index = score_keys.index(best_key)

        for idx, eval_item in enumerate(candidates):
            task = self.instance.tasks[eval_item.task_id]
            remaining_nb = max(task.optimistic_nonbatch_from[eval_item.idx], 1)
            duration = eval_item.finish - eval_item.start
            q_slack = eval_item.upper_bound - eval_item.start
            delivery_gap = task.delivery_time - eval_item.est_final
            action_mask[idx] = 1.0
            candidate_keys.append(f"{eval_item.task_id}:{task.processes[eval_item.idx].seq}:{eval_item.machine_id}")
            candidate_features[idx] = np.asarray(
                [
                    task.weight / 20.0,
                    (task.weight / remaining_nb),
                    eval_item.idx / max(len(task.processes) - 1, 1),
                    1.0 if eval_item.started else 0.0,
                    1.0 if eval_item.same_family else 0.0,
                    1.0 if eval_item.zero_setup else 0.0,
                    eval_item.setup_time / 120.0,
                    (eval_item.start - min_start) / max(self.lookahead, 1),
                    duration / 240.0,
                    (self.instance.horizon - eval_item.finish) / horizon_span,
                    (self.instance.horizon - eval_item.est_final) / horizon_span,
                    min(q_slack, horizon_span) / horizon_span,
                    delivery_gap / max(horizon_span, 1),
                    task.priority / 10.0,
                    eval_item.option_priority / 10.0,
                    remaining_nb / max(len(task.processes), 1),
                ],
                dtype=np.float32,
            )

        observation = DispatchObservation(
            global_features=global_features,
            candidate_features=candidate_features,
            action_mask=action_mask,
            heuristic_index=heuristic_index,
            candidate_keys=candidate_keys,
        )
        return observation, candidates, min_start

    def step_phase1(self, choice: CandidateEval, min_start: int) -> float:
        setup_before = self.setup_count
        self.record_schedule(choice)
        self.task_status[choice.task_id] = "active"
        self.advance_batch_prefix(choice.task_id, respect_horizon=True)
        gained_weight, _ = self._harvest_completed_weight()

        reward = gained_weight * self.reward_weight_scale
        reward -= (self.setup_count - setup_before) * self.reward_setup_penalty
        reward -= max(choice.start - min_start, 0) / max(self.lookahead, 1) * self.reward_delay_penalty
        reward -= max(choice.est_final - self.instance.horizon, 0) / 60.0 * self.reward_late_penalty
        if choice.same_family:
            reward += self.reward_same_family_bonus
        if choice.zero_setup:
            reward += self.reward_zero_setup_bonus
        return reward

    def phase1_metrics(self) -> dict[str, float]:
        return {
            "completed_weight": round(self.completed_weight, 3),
            "completed_tasks": float(self.completed_tasks),
            "setup_count_positive": float(self.setup_count),
        }


class DispatchPPOEnv:
    def __init__(
        self,
        full_instance: InstanceData,
        setup_store: SetupRowStore,
        task_limit: int = 256,
        task_sampling: str = "mixed",
        max_candidates: int = 32,
        lookahead: int = 70,
        start_guard: int = 360,
        decision_budget: int = 512,
        seed: int = 0,
        reward_weight_scale: float = 100.0,
        reward_setup_penalty: float = 6.0,
        reward_delay_penalty: float = 0.2,
        reward_late_penalty: float = 12.0,
        reward_same_family_bonus: float = 0.5,
        reward_zero_setup_bonus: float = 0.75,
    ) -> None:
        self.full_instance = full_instance
        self.setup_store = setup_store
        self.task_limit = task_limit
        self.task_sampling = task_sampling
        self.max_candidates = max_candidates
        self.lookahead = lookahead
        self.start_guard = start_guard
        self.decision_budget = decision_budget
        self.rng = random.Random(seed)
        self.reward_weight_scale = reward_weight_scale
        self.reward_setup_penalty = reward_setup_penalty
        self.reward_delay_penalty = reward_delay_penalty
        self.reward_late_penalty = reward_late_penalty
        self.reward_same_family_bonus = reward_same_family_bonus
        self.reward_zero_setup_bonus = reward_zero_setup_bonus

        self.scheduler: Optional[Phase1DispatchScheduler] = None
        self.current_candidates: list[CandidateEval] = []
        self.current_min_start = 0
        self.decision_count = 0

    def reset(self) -> DispatchObservation:
        task_ids = sample_task_ids(self.full_instance, self.task_limit, self.task_sampling, self.rng)
        instance = subset_instance(self.full_instance, task_ids)
        self.scheduler = Phase1DispatchScheduler(
            instance,
            self.setup_store,
            lookahead=self.lookahead,
            start_guard=self.start_guard,
            max_candidates=self.max_candidates,
            reward_weight_scale=self.reward_weight_scale,
            reward_setup_penalty=self.reward_setup_penalty,
            reward_delay_penalty=self.reward_delay_penalty,
            reward_late_penalty=self.reward_late_penalty,
            reward_same_family_bonus=self.reward_same_family_bonus,
            reward_zero_setup_bonus=self.reward_zero_setup_bonus,
        )
        self.scheduler.bootstrap()
        self.decision_count = 0
        feasible_candidates = self.scheduler.collect_phase1_candidates()
        observation, candidates, min_start = self.scheduler.build_observation(feasible_candidates)
        self.current_candidates = candidates
        self.current_min_start = min_start
        return observation

    def step(self, action_index: int) -> tuple[Optional[DispatchObservation], float, bool, dict[str, float]]:
        if self.scheduler is None:
            raise RuntimeError("Environment must be reset before stepping")
        if action_index < 0 or action_index >= len(self.current_candidates):
            raise IndexError(f"Invalid action index: {action_index}")

        choice = self.current_candidates[action_index]
        reward = self.scheduler.step_phase1(choice, self.current_min_start)
        self.decision_count += 1

        done = self.decision_count >= self.decision_budget
        if not done:
            feasible_candidates = self.scheduler.collect_phase1_candidates()
            if feasible_candidates:
                observation, candidates, min_start = self.scheduler.build_observation(feasible_candidates)
                self.current_candidates = candidates
                self.current_min_start = min_start
                info = self.scheduler.phase1_metrics()
                info["decision_count"] = float(self.decision_count)
                return observation, reward, False, info
            done = True

        self.current_candidates = []
        info = self.scheduler.phase1_metrics()
        info["decision_count"] = float(self.decision_count)
        return None, reward, True, info

    def heuristic_action(self, observation: DispatchObservation) -> int:
        valid = int(observation.action_mask.sum())
        return min(observation.heuristic_index, max(valid - 1, 0))

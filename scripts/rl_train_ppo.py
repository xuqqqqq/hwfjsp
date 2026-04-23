#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rl_phase1_env import DispatchObservation, DispatchPPOEnv
from rl_relaxed_solver import SetupRowStore, build_instance, detect_input_json


@dataclass
class PPOBatch:
    global_features: torch.Tensor
    candidate_features: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class CandidatePolicyValueNet(nn.Module):
    def __init__(self, global_dim: int, candidate_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.candidate_mlp = nn.Sequential(
            nn.Linear(global_dim + candidate_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logit_head = nn.Linear(hidden_dim, 1)
        self.value_mlp = nn.Sequential(
            nn.Linear(global_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        global_features: torch.Tensor,
        candidate_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_global = global_features.unsqueeze(1).expand(
            -1,
            candidate_features.size(1),
            -1,
        )
        joint = torch.cat([expanded_global, candidate_features], dim=-1)
        cand_hidden = self.candidate_mlp(joint)
        logits = self.logit_head(cand_hidden).squeeze(-1)
        logits = logits.masked_fill(action_mask <= 0, -1e9)

        valid_weights = action_mask.unsqueeze(-1)
        pooled = (cand_hidden * valid_weights).sum(dim=1) / valid_weights.sum(dim=1).clamp_min(1.0)
        value_input = torch.cat([global_features, pooled], dim=-1)
        values = self.value_mlp(value_input).squeeze(-1)
        return logits, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the phase-1 dispatch environment")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--instance-cache", type=Path, default=Path("cache/actual_scale.pkl"))
    parser.add_argument("--setup-db", type=Path, default=Path("cache/actual_scale.sqlite"))
    parser.add_argument("--horizon-override", type=int, default=24480)
    parser.add_argument("--task-limit", type=int, default=256)
    parser.add_argument("--task-sampling", choices=["random", "weight", "earliest", "mixed"], default="mixed")
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--lookahead", type=int, default=70)
    parser.add_argument("--start-guard", type=int, default=360)
    parser.add_argument("--decision-budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl_runs"))
    parser.add_argument("--checkpoint-name", type=str, default="ppo_phase1_latest.pt")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny train loop for validation")
    return parser.parse_args()


def tensorize_observation(observation: DispatchObservation, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_tensor = torch.as_tensor(observation.global_features, dtype=torch.float32, device=device)
    candidate_tensor = torch.as_tensor(observation.candidate_features, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(observation.action_mask, dtype=torch.float32, device=device)
    return global_tensor, candidate_tensor, mask_tensor


def select_action(
    model: CandidatePolicyValueNet,
    observation: DispatchObservation,
    device: torch.device,
) -> tuple[int, float, float]:
    global_tensor, candidate_tensor, mask_tensor = tensorize_observation(observation, device)
    logits, value = model(
        global_tensor.unsqueeze(0),
        candidate_tensor.unsqueeze(0),
        mask_tensor.unsqueeze(0),
    )
    dist = Categorical(logits=logits)
    action = dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def select_greedy_action(
    model: CandidatePolicyValueNet,
    observation: DispatchObservation,
    device: torch.device,
) -> int:
    global_tensor, candidate_tensor, mask_tensor = tensorize_observation(observation, device)
    logits, _ = model(
        global_tensor.unsqueeze(0),
        candidate_tensor.unsqueeze(0),
        mask_tensor.unsqueeze(0),
    )
    return int(torch.argmax(logits, dim=-1).item())


def rollout_policy(
    env: DispatchPPOEnv,
    model: CandidatePolicyValueNet,
    device: torch.device,
    target_steps: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[PPOBatch, list[dict[str, float]]]:
    observations_g: list[np.ndarray] = []
    observations_c: list[np.ndarray] = []
    observations_m: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    dones: list[bool] = []
    episode_infos: list[dict[str, float]] = []

    obs = env.reset()
    current_obs = obs
    while len(actions) < target_steps:
        action, log_prob, value = select_action(model, current_obs, device)
        next_obs, reward, done, info = env.step(action)

        observations_g.append(current_obs.global_features.copy())
        observations_c.append(current_obs.candidate_features.copy())
        observations_m.append(current_obs.action_mask.copy())
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(done)

        if done:
            episode_infos.append(info)
            current_obs = env.reset()
        else:
            current_obs = next_obs

    with torch.no_grad():
        if dones and not dones[-1]:
            global_tensor, candidate_tensor, mask_tensor = tensorize_observation(current_obs, device)
            _, value_tensor = model(
                global_tensor.unsqueeze(0),
                candidate_tensor.unsqueeze(0),
                mask_tensor.unsqueeze(0),
            )
            bootstrap_value = float(value_tensor.item())
        else:
            bootstrap_value = 0.0

    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_adv = 0.0
    next_value = bootstrap_value
    for idx in reversed(range(len(rewards))):
        mask = 0.0 if dones[idx] else 1.0
        delta = rewards[idx] + gamma * next_value * mask - values[idx]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[idx] = last_adv
        next_value = values[idx]
    returns = advantages + np.asarray(values, dtype=np.float32)

    batch = PPOBatch(
        global_features=torch.as_tensor(np.asarray(observations_g), dtype=torch.float32, device=device),
        candidate_features=torch.as_tensor(np.asarray(observations_c), dtype=torch.float32, device=device),
        action_mask=torch.as_tensor(np.asarray(observations_m), dtype=torch.float32, device=device),
        actions=torch.as_tensor(actions, dtype=torch.long, device=device),
        log_probs=torch.as_tensor(log_probs, dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
    )
    return batch, episode_infos


def ppo_update(
    model: CandidatePolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    ppo_epochs: int,
    mini_batch_size: int,
    clip_ratio: float,
    entropy_coef: float,
    value_coef: float,
) -> dict[str, float]:
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    batch = PPOBatch(
        global_features=batch.global_features,
        candidate_features=batch.candidate_features,
        action_mask=batch.action_mask,
        actions=batch.actions,
        log_probs=batch.log_probs,
        returns=batch.returns,
        advantages=advantages,
    )

    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    total_steps = batch.actions.size(0)
    indices = np.arange(total_steps)

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, total_steps, mini_batch_size):
            batch_idx = indices[start : start + mini_batch_size]
            logits, values = model(
                batch.global_features[batch_idx],
                batch.candidate_features[batch_idx],
                batch.action_mask[batch_idx],
            )
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch.actions[batch_idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - batch.log_probs[batch_idx])
            unclipped = ratio * batch.advantages[batch_idx]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch.advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch.returns[batch_idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics["loss"] += float(loss.item())
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())

    denom = max((ppo_epochs * math.ceil(total_steps / mini_batch_size)), 1)
    return {key: round(value / denom, 6) for key, value in metrics.items()}


def evaluate_policy(
    env: DispatchPPOEnv,
    model: CandidatePolicyValueNet,
    device: torch.device,
    episodes: int = 3,
) -> dict[str, float]:
    weights: list[float] = []
    setups: list[float] = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        last_info: dict[str, float] = {}
        while not done:
            action = select_greedy_action(model, obs, device)
            next_obs, _reward, done, last_info = env.step(action)
            obs = next_obs if next_obs is not None else obs
        weights.append(last_info.get("completed_weight", 0.0))
        setups.append(last_info.get("setup_count_positive", 0.0))
    return {
        "eval_completed_weight": round(float(np.mean(weights)), 3),
        "eval_setup_count": round(float(np.mean(setups)), 3),
    }


def heuristic_rollout(env: DispatchPPOEnv, episodes: int = 3) -> dict[str, float]:
    weights: list[float] = []
    setups: list[float] = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        last_info: dict[str, float] = {}
        while not done:
            action = env.heuristic_action(obs)
            next_obs, _reward, done, last_info = env.step(action)
            obs = next_obs if next_obs is not None else obs
        weights.append(last_info.get("completed_weight", 0.0))
        setups.append(last_info.get("setup_count_positive", 0.0))
    return {
        "heuristic_completed_weight": round(float(np.mean(weights)), 3),
        "heuristic_setup_count": round(float(np.mean(setups)), 3),
    }


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.task_limit = min(args.task_limit, 64)
        args.decision_budget = min(args.decision_budget, 64)
        args.updates = min(args.updates, 1)
        args.steps_per_update = min(args.steps_per_update, 64)
        args.ppo_epochs = min(args.ppo_epochs, 2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = args.root.resolve()
    input_path = args.input.resolve() if args.input else detect_input_json(root)
    instance_cache = (root / args.instance_cache).resolve() if not args.instance_cache.is_absolute() else args.instance_cache
    setup_db = (root / args.setup_db).resolve() if not args.setup_db.is_absolute() else args.setup_db
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    checkpoint_path = output_dir / args.checkpoint_name

    instance = build_instance(root, input_path, instance_cache, force=False)
    if args.horizon_override is not None:
        instance.horizon = int(args.horizon_override)
    setup_store = SetupRowStore(setup_db)
    setup_store.ensure(input_path, force=False)

    device = torch.device(args.device)
    env = DispatchPPOEnv(
        instance,
        setup_store,
        task_limit=args.task_limit,
        task_sampling=args.task_sampling,
        max_candidates=args.max_candidates,
        lookahead=args.lookahead,
        start_guard=args.start_guard,
        decision_budget=args.decision_budget,
        seed=args.seed,
    )

    obs = env.reset()
    global_dim = int(obs.global_features.shape[0])
    candidate_dim = int(obs.candidate_features.shape[1])
    model = CandidatePolicyValueNet(global_dim, candidate_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    heuristic_metrics = heuristic_rollout(env, episodes=2 if args.smoke else 3)
    print(json.dumps({"baseline": heuristic_metrics}, ensure_ascii=False, indent=2), flush=True)

    history: list[dict[str, float]] = []
    try:
        for update_idx in range(1, args.updates + 1):
            batch, episode_infos = rollout_policy(
                env,
                model,
                device,
                target_steps=args.steps_per_update,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            train_metrics = ppo_update(
                model,
                optimizer,
                batch,
                ppo_epochs=args.ppo_epochs,
                mini_batch_size=args.mini_batch_size,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
            )
            eval_metrics = evaluate_policy(env, model, device, episodes=2 if args.smoke else 3)
            episode_weight = float(np.mean([info.get("completed_weight", 0.0) for info in episode_infos])) if episode_infos else 0.0
            episode_setup = float(np.mean([info.get("setup_count_positive", 0.0) for info in episode_infos])) if episode_infos else 0.0

            record = {
                "update": float(update_idx),
                "rollout_completed_weight": round(episode_weight, 3),
                "rollout_setup_count": round(episode_setup, 3),
                **train_metrics,
                **eval_metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)

        torch.save(
            {
                "model_state": model.state_dict(),
                "config": vars(args),
                "history": history,
                "baseline": heuristic_metrics,
            },
            checkpoint_path,
        )
        with (output_dir / "ppo_phase1_history.json").open("w", encoding="utf-8") as fh:
            json.dump({"baseline": heuristic_metrics, "history": history}, fh, ensure_ascii=False, indent=2)
        print(f"[saved] checkpoint: {checkpoint_path}", flush=True)
        return 0
    finally:
        setup_store.close()


if __name__ == "__main__":
    raise SystemExit(main())

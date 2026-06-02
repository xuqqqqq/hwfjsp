#!/usr/bin/env python3
"""训练算例级策略网络。

训练目标是学习“算例特征 -> 安全参数/规则模板”的评分函数。网络不直接派工，
只在 rl_policy_recommender.py 定义的候选模板之间打分；硬约束仍由求解器和
校验器负责。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数。"""

    parser = argparse.ArgumentParser(description="Train a case-level FJSP policy network.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--track", choices=("finite", "relaxed"), default="finite")
    parser.add_argument(
        "--policy-state",
        type=Path,
        default=Path("outputs/rl_policy_recommender/policy_state.json"),
        help="读取 bandit/supervised 累计经验的状态文件。",
    )
    parser.add_argument(
        "--experience-source",
        action="append",
        type=Path,
        default=[],
        help="额外读取的 experience.jsonl、final_report.json 或目录，可重复传入。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rl_policy_recommender/policy_network.json"),
        help="导出的策略网络 JSON 文件。",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--reward-setup-penalty", type=float, default=0.2)
    parser.add_argument("--reward-error-penalty", type=float, default=5000.0)
    parser.add_argument("--reward-time-penalty", type=float, default=0.02)
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    """把相对路径解析到项目根目录。"""

    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_recommender(root: Path):
    """从文件路径加载 rl_policy_recommender 模块。"""

    module_path = root / "scripts" / "rl_policy_recommender.py"
    spec = importlib.util.spec_from_file_location("rl_policy_recommender_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load recommender from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def to_tensor(values: list[list[float]]) -> torch.Tensor:
    """把二维列表转换为 float32 Tensor。"""

    return torch.tensor(values, dtype=torch.float32)


class PolicyNet(torch.nn.Module):
    """小型 MLP 模板评分网络。"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc1(features))
        return self.fc2(hidden)


def collect_examples(
    recommender: Any,
    *,
    state: dict[str, Any],
    sources: list[Path],
    setup_penalty: float,
    error_penalty: float,
    time_penalty: float,
) -> list[dict[str, Any]]:
    """从状态文件和历史报告中收集训练样本。"""

    examples: list[dict[str, Any]] = []
    reports = recommender.load_experiences(state, sources)
    for report in reports:
        if not isinstance(report, dict):
            continue
        features = report.get("features")
        recommendation = report.get("recommendation", {}) or {}
        candidate_id = recommendation.get("candidate_id")
        if not isinstance(features, dict) or not candidate_id:
            continue
        reward_details = report.get("reward_details")
        if not reward_details:
            reward_details = recommender.reward_from_report(
                report,
                setup_penalty=setup_penalty,
                error_penalty=error_penalty,
                time_penalty=time_penalty,
            )
        if not reward_details:
            continue
        examples.append(
            {
                "features": features,
                "candidate_id": str(candidate_id),
                "reward": float(reward_details["reward"]),
                "metrics": {
                    "weight": reward_details.get("weight"),
                    "setup": reward_details.get("setup"),
                    "error_count": reward_details.get("error_count"),
                },
            }
        )
    return examples


def export_model(
    path: Path,
    *,
    model: PolicyNet,
    recommender: Any,
    candidate_ids: list[str],
    reward_center: float,
    reward_scale: float,
    training_summary: dict[str, Any],
) -> None:
    """导出推荐器可直接读取的 JSON 模型。"""

    payload = {
        "model_type": "mlp_template_score",
        "feature_keys": recommender.FEATURE_VECTOR_KEYS,
        "feature_scales": recommender.FEATURE_SCALES,
        "candidate_ids": candidate_ids,
        "reward_center": reward_center,
        "reward_scale": reward_scale,
        "weights": {
            "w1": model.fc1.weight.detach().cpu().tolist(),
            "b1": model.fc1.bias.detach().cpu().tolist(),
            "w2": model.fc2.weight.detach().cpu().tolist(),
            "b2": model.fc2.bias.detach().cpu().tolist(),
        },
        "training_summary": training_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """训练主流程。"""

    args = parse_args()
    root = args.root.resolve()
    policy_state_path = resolve_path(root, args.policy_state)
    output_path = resolve_path(root, args.output)
    sources = [resolve_path(root, path) for path in args.experience_source]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    recommender = load_recommender(root)
    state = recommender.load_policy_state(policy_state_path)
    examples = collect_examples(
        recommender,
        state=state,
        sources=sources,
        setup_penalty=args.reward_setup_penalty,
        error_penalty=args.reward_error_penalty,
        time_penalty=args.reward_time_penalty,
    )
    if not examples:
        raise SystemExit("no training examples found")

    first_features = examples[0]["features"]
    candidates = recommender.candidate_policy_library(first_features, args.track, target_setup=600)
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    examples = [item for item in examples if item["candidate_id"] in candidate_index]
    if not examples:
        raise SystemExit("no examples match candidate library")

    rewards = [float(item["reward"]) for item in examples]
    reward_center = sum(rewards) / len(rewards)
    variance = sum((reward - reward_center) ** 2 for reward in rewards) / max(len(rewards), 1)
    reward_scale = max(100.0, variance ** 0.5)

    x_values = [recommender.feature_vector(item["features"]) for item in examples]
    action_values = [candidate_index[item["candidate_id"]] for item in examples]
    y_values = [(float(item["reward"]) - reward_center) / reward_scale for item in examples]

    x_tensor = to_tensor(x_values)
    actions = torch.tensor(action_values, dtype=torch.long)
    targets = torch.tensor(y_values, dtype=torch.float32)

    model = PolicyNet(
        input_size=len(recommender.FEATURE_VECTOR_KEYS),
        hidden_size=max(4, args.hidden_size),
        output_size=len(candidate_ids),
    )
    with torch.no_grad():
        torch.nn.init.normal_(model.fc1.weight, mean=0.0, std=0.05)
        torch.nn.init.zeros_(model.fc1.bias)
        torch.nn.init.zeros_(model.fc2.weight)
        prior_scores = []
        for candidate in candidates:
            arm = state.get("arms", {}).get(candidate["candidate_id"], {})
            if arm.get("count"):
                prior_reward = float(arm.get("reward_mean", candidate.get("candidate_prior_reward", reward_center)))
            else:
                prior_reward = float(candidate.get("candidate_prior_reward", reward_center))
            prior_scores.append((prior_reward - reward_center) / reward_scale)
        model.fc2.bias.copy_(torch.tensor(prior_scores, dtype=torch.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    last_loss = 0.0
    for _epoch in range(max(1, args.epochs)):
        optimizer.zero_grad()
        scores = model(x_tensor)
        predicted = scores.gather(1, actions.view(-1, 1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(predicted, targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        train_scores = model(x_tensor)
        chosen = train_scores.argmax(dim=1).tolist()
    correct = sum(1 for index, action in enumerate(action_values) if chosen[index] == action)
    best_by_reward = max(examples, key=lambda item: float(item["reward"]))

    training_summary = {
        "example_count": len(examples),
        "candidate_count": len(candidate_ids),
        "epochs": args.epochs,
        "hidden_size": max(4, args.hidden_size),
        "final_loss": round(last_loss, 8),
        "train_action_match_rate": round(correct / len(examples), 6),
        "best_observed_candidate": best_by_reward["candidate_id"],
        "best_observed_reward": round(float(best_by_reward["reward"]), 6),
        "best_observed_metrics": best_by_reward.get("metrics", {}),
        "policy_state": str(policy_state_path),
    }
    export_model(
        output_path,
        model=model,
        recommender=recommender,
        candidate_ids=candidate_ids,
        reward_center=reward_center,
        reward_scale=reward_scale,
        training_summary=training_summary,
    )
    print(json.dumps({"output": str(output_path), "training_summary": training_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

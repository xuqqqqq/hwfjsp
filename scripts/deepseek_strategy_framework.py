#!/usr/bin/env python3
"""DeepSeek 驱动的策略模块自迭代框架。

与 `deepseek_solver_framework.py` 不同，本框架不再让 LLM 生成完整求解器。
LLM 只能生成 `deepseek_strategy.py`，固定约束内核负责：

- 输入解析与路径实例构建；
- 可行动作过滤；
- 两阶段调度主循环；
- 状态提交和解文件写出；
- 有限组批校验。

这样保留了 LLM 自我迭代规则的能力，同时避免它改坏 qtime、维修、转运、
有限组批等硬约束。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable

from deepseek_solver_framework import call_deepseek, extract_python_code, load_documents


DEFAULT_DOCS = (
    "问题描述文档.md",
    "输入输出结构.md",
    "rl1.md",
)


FORBIDDEN_IMPORT_ROOTS = {
    "os",
    "sys",
    "pathlib",
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "http",
    "rl_relaxed_solver",
    "validate_batch_solution",
    "strategy_kernel_solver",
    "deepseek_solver_framework",
}

FORBIDDEN_CALL_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "input",
}


REQUIRED_FUNCTIONS = (
    "describe_rule_changes",
    "get_config",
    "score_path",
    "score_phase1",
    "score_phase2",
)


CANDIDATE_MODES = (
    "candidate_pool_operator",
    "direct_action_operator",
    "batch_group_operator",
    "phase_window_operator",
    "new_dispatch_rule",
    "new_completion_rule",
    "rule_prune_ablation",
    "operator_crossover_prune",
    "guarded_parameter_mutation",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Iterate DeepSeek-generated strategy modules under a fixed FJSP kernel."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=Path("data/data1/测试算例1.json"))
    parser.add_argument("--track", choices=("finite", "relaxed"), default="finite")
    parser.add_argument(
        "--strategy-path",
        type=Path,
        default=Path("deepseek_strategy.py"),
        help="根目录下当前策略模块输出路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/deepseek_strategy_framework"),
    )
    parser.add_argument("--doc", action="append", default=[])
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--wire-api",
        choices=("chat", "responses"),
        default="chat",
        help=(
            "LLM 接口协议。chat 使用 /chat/completions；responses 使用 "
            "/responses，适配 Codex 类代理服务。"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.18)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=180,
        help="单次 LLM HTTP 请求超时时间，单位秒。",
    )
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument(
        "--candidates-per-iter",
        type=int,
        default=4,
        help="每轮生成的策略候选数。候选会使用不同探索角色。默认 4。",
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=4,
        help="反馈给下一轮交叉/变异 prompt 的精英候选数量。",
    )
    parser.add_argument("--solver-timeout", type=int, default=420)
    parser.add_argument("--setup-penalty", type=float, default=0.2)
    parser.add_argument("--target-weight", type=float, default=18800.0)
    parser.add_argument("--target-setup", type=int, default=900)
    parser.add_argument(
        "--no-seed-baseline",
        action="store_true",
        help="不先运行默认策略基线。默认会先建立合法基线，防止 LLM 退化策略被误认为 best。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    """把相对路径解析到项目根目录。"""

    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON；文件不存在或格式错误时返回空 dict。"""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    """写出 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    """追加实验经验。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n\n")


def check_strategy_safety(code: str) -> list[str]:
    """静态检查策略文件，防止 LLM 越权改内核或读写文件。"""

    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        issues.append(f"strategy syntax error: {exc}")
        return issues
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    issues.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                issues.append(f"forbidden import: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                issues.append(f"forbidden call: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute):
                root = node.func.attr
                if root in FORBIDDEN_CALL_NAMES:
                    issues.append(f"forbidden call: {root}()")
    for name in REQUIRED_FUNCTIONS:
        if not re.search(rf"def\s+{name}\s*\(", code):
            issues.append(f"missing required strategy function: {name}")
    if re.search(r"def\s+describe_rule_changes\s*\(", code):
        for key in (
            "added_rules",
            "removed_rules",
            "kept_rules",
            "changed_parameters",
            "rule_suggestions",
            "expected_effect",
        ):
            if key not in code:
                issues.append(f"describe_rule_changes missing audit key: {key}")
    return issues


def metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全读取浮点指标。"""

    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def metric_int(metrics: dict[str, Any], key: str, default: int = 0) -> int:
    """安全读取整数指标。"""

    try:
        return int(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def is_complete_legal(summary: dict[str, Any]) -> bool:
    """判断固定内核输出是否为完整合法解。"""

    metrics = summary.get("validation_metrics", {})
    return (
        int(summary.get("error_count", 999999)) == 0
        and metric_int(metrics, "fully_scheduled_tasks") == metric_int(metrics, "total_tasks")
    )


def score_summary(
    summary: dict[str, Any],
    setup_penalty: float,
    *,
    target_weight: float = 0.0,
    target_setup: int = 0,
) -> float:
    """合法候选的目标分；非法候选为负无穷。

    默认模式为产量减 setup 惩罚。若传入目标产量和目标 setup，则使用更贴近
    “185xx/500 左右”的分层评分：优先保留 setup 接近目标的候选，再在同一
    setup 带内回补产量，避免 18680+/640+ 的高产高切换解长期支配经验池。
    """

    if not is_complete_legal(summary):
        return float("-inf")
    metrics = summary.get("validation_metrics", {})
    weight = metric_float(metrics, "completed_weight_within_horizon")
    setup = metric_int(metrics, "setup_count_positive")
    if target_weight > 0 and target_setup > 0:
        setup_over = max(0, setup - target_setup)
        deficit = max(0.0, target_weight - weight)
        if weight >= target_weight and setup <= target_setup + 120:
            tier = 100000.0
        elif weight >= target_weight - 500 and setup <= target_setup + 120:
            tier = 90000.0
        elif weight >= target_weight:
            tier = 80000.0
        else:
            tier = 70000.0
        return tier + weight * 0.2 - deficit * 4.0 - setup_over * 100.0 - setup_penalty * setup
    return weight - setup_penalty * setup


def candidate_mode(iteration: int, candidate_index: int) -> str:
    """为候选分配探索角色，避免所有候选沿同一方向退化。"""

    offset = iteration % len(CANDIDATE_MODES)
    return CANDIDATE_MODES[(candidate_index + offset) % len(CANDIDATE_MODES)]


def summarize_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """提取 LLM 反思需要的核心指标。"""

    metrics = summary.get("validation_metrics", {})
    config = summary.get("config", {})
    return {
        "strategy_name": summary.get("strategy_name"),
        "candidate": summary.get("candidate"),
        "rule_changes": summary.get("rule_changes"),
        "completed_weight_within_horizon": metrics.get("completed_weight_within_horizon"),
        "setup_count_positive": metrics.get("setup_count_positive"),
        "fully_scheduled_tasks": metrics.get("fully_scheduled_tasks"),
        "total_tasks": metrics.get("total_tasks"),
        "error_count": summary.get("error_count"),
        "rule_change_quality_warning": summary.get("rule_change_quality_warning"),
        "diversity_warning": summary.get("diversity_warning"),
        "key_config": {
            key: config.get(key)
            for key in (
                "lookahead",
                "start_guard",
                "score_weight",
                "score_density",
                "score_family",
                "score_progress",
                "score_zero_setup",
                "score_setup_fixed",
                "score_setup_per",
                "phase2_started",
                "phase2_density",
                "phase2_family",
                "phase2_progress",
                "phase2_zero_setup",
                "phase2_setup_fixed",
                "phase2_allow_unstarted",
                "batch_group_wait",
                "batch_group_mixed_time",
                "batch_group_any_time",
            )
            if key in config
        },
    }


def audit_rule_changes(rule_changes: Any) -> list[str]:
    """检查策略是否留下可复盘的规则/参数演化说明。"""

    if not isinstance(rule_changes, dict):
        return ["describe_rule_changes() must return a dict"]
    issues: list[str] = []
    added = rule_changes.get("added_rules") or []
    removed = rule_changes.get("removed_rules") or []
    kept = rule_changes.get("kept_rules") or []
    changed = rule_changes.get("changed_parameters") or {}
    suggestions = rule_changes.get("rule_suggestions") or []
    expected = str(rule_changes.get("expected_effect") or "").strip()
    if not isinstance(added, list):
        issues.append("added_rules must be a list")
        added = []
    if not isinstance(removed, list):
        issues.append("removed_rules must be a list")
        removed = []
    if not isinstance(kept, list):
        issues.append("kept_rules must be a list")
        kept = []
    if not isinstance(changed, dict):
        issues.append("changed_parameters must be a dict")
        changed = {}
    if not isinstance(suggestions, list):
        issues.append("rule_suggestions must be a list")
        suggestions = []
    if not added and not removed and not changed:
        issues.append(
            "parameter-only candidates are allowed, but added_rules/removed_rules/changed_parameters cannot all be empty"
        )
    if not added and not removed and not suggestions:
        issues.append(
            "parameter-only candidates must still propose rule_suggestions for future rule-level experiments"
        )
    if not expected:
        issues.append("expected_effect is required")
    removed_tags = rule_audit_tags(removed)
    active_tags = rule_audit_tags(added) | rule_audit_tags(kept)
    conflicts = sorted(removed_tags & active_tags)
    if conflicts:
        issues.append(
            "rule audit is internally inconsistent; removed rules also appear in added/kept: "
            + ", ".join(conflicts)
        )
    return issues


def rule_audit_tags(items: list[Any]) -> set[str]:
    """把规则审计文本映射成粗粒度标签，用于发现自相矛盾的增删记录。"""

    tags: set[str] = set()
    for item in items:
        text = str(item).lower()
        if "score_path" in text:
            if "返回none" in text or "自定义路径" in text or "所有自定义" in text:
                tags.add("score_path_custom")
            if "组批" in text:
                tags.add("score_path_batch")
            if "紧迫" in text:
                tags.add("score_path_urgency")
            if "负载" in text:
                tags.add("score_path_balance")
        if "score_phase1" in text:
            if "密度" in text:
                tags.add("score_phase1_density")
            if "进度" in text:
                tags.add("score_phase1_progress")
            if "组批" in text:
                tags.add("score_phase1_batch")
            if "负载" in text:
                tags.add("score_phase1_balance")
        if "score_phase2" in text:
            if "路径剩余密度" in text:
                tags.add("score_phase2_path_density")
            elif "密度" in text:
                tags.add("score_phase2_density")
            if "进度" in text:
                tags.add("score_phase2_progress")
            if "上界" in text:
                tags.add("score_phase2_upper_bound")
            if "紧迫" in text:
                tags.add("score_phase2_urgency")
            if "组批" in text:
                tags.add("score_phase2_batch")
            if "机器负载" in text or "负载均衡" in text:
                tags.add("score_phase2_machine_balance")
    return tags


def population_brief(
    population: list[dict[str, Any]],
    *,
    elite_size: int,
) -> str:
    """把当前种群压缩成下一轮可使用的反思材料。"""

    if not population:
        return "暂无候选种群。"
    ranked = sorted(population, key=lambda item: item.get("score", float("-inf")), reverse=True)
    elite_blocks: list[str] = []
    for rank, item in enumerate(ranked[:elite_size], start=1):
        candidate_path = Path(str(item.get("candidate_path", "")))
        code = ""
        if candidate_path.exists():
            code = candidate_path.read_text(encoding="utf-8", errors="replace")[-9000:]
        elite_blocks.append(
            textwrap.dedent(
                f"""
                ### Elite {rank}
                score: {item.get("score")}
                metrics:
                ```json
                {json.dumps(summarize_metrics(item.get("summary", {})), ensure_ascii=False, indent=2)}
                ```
                code tail:
                ```python
                {code}
                ```
                """
            ).strip()
        )
    failed = [
        summarize_metrics(item.get("summary", {}))
        for item in ranked[elite_size : elite_size + 8]
    ]
    low_setup = sorted(
        [
            item
            for item in population
            if is_complete_legal(item.get("summary", {}))
        ],
        key=lambda item: (
            metric_int(item.get("summary", {}).get("validation_metrics", {}), "setup_count_positive", 999999),
            -metric_float(item.get("summary", {}).get("validation_metrics", {}), "completed_weight_within_horizon"),
        ),
    )
    low_setup_summary = [
        summarize_metrics(item.get("summary", {}))
        for item in low_setup[: max(4, elite_size)]
    ]
    return (
        "\n\n".join(elite_blocks)
        + "\n\n### 低切换候选摘要\n"
        + json.dumps(low_setup_summary, ensure_ascii=False, indent=2)
        + "\n\n### 其他候选摘要\n"
        + json.dumps(failed, ensure_ascii=False, indent=2)
    )


def select_population(
    population: list[dict[str, Any]],
    *,
    elite_size: int,
    target_weight: float,
    target_setup: int,
) -> list[dict[str, Any]]:
    """保留多目标候选，避免单一综合分压制低切换经验。

    DeepSeek 生成的策略常会围绕一个高产高切换局部结构反复微调。种群选择
    因此分三条通道：综合分最高、产量达标且 setup 最低、产量在目标带附近且
    setup 最接近目标。这样下一轮 prompt 同时能看到高产解和低切换经验。
    """

    complete = [
        item
        for item in population
        if isinstance(item, dict) and is_complete_legal(item.get("summary", {}))
    ]
    invalid = [item for item in population if item not in complete]
    score_ranked = sorted(
        complete,
        key=lambda item: safe_number(item.get("score"), float("-inf")),
        reverse=True,
    )
    setup_ranked = sorted(
        [
            item
            for item in complete
            if metric_float(
                item.get("summary", {}).get("validation_metrics", {}),
                "completed_weight_within_horizon",
            )
            >= target_weight
        ],
        key=lambda item: (
            metric_int(item.get("summary", {}).get("validation_metrics", {}), "setup_count_positive", 999999),
            -metric_float(item.get("summary", {}).get("validation_metrics", {}), "completed_weight_within_horizon"),
        ),
    )
    band_ranked = sorted(
        [
            item
            for item in complete
            if target_weight - 250
            <= metric_float(item.get("summary", {}).get("validation_metrics", {}), "completed_weight_within_horizon")
            <= target_weight + 350
        ],
        key=lambda item: (
            abs(
                metric_int(
                    item.get("summary", {}).get("validation_metrics", {}),
                    "setup_count_positive",
                    999999,
                )
                - target_setup
            ),
            metric_int(item.get("summary", {}).get("validation_metrics", {}), "setup_count_positive", 999999),
            -metric_float(item.get("summary", {}).get("validation_metrics", {}), "completed_weight_within_horizon"),
        ),
    )
    low_setup_any = sorted(
        complete,
        key=lambda item: (
            metric_int(item.get("summary", {}).get("validation_metrics", {}), "setup_count_positive", 999999),
            -metric_float(item.get("summary", {}).get("validation_metrics", {}), "completed_weight_within_horizon"),
        ),
    )

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in (
        score_ranked[: max(elite_size * 2, 8)],
        setup_ranked[: max(elite_size * 2, 8)],
        band_ranked[: max(elite_size * 2, 8)],
        low_setup_any[: max(elite_size, 4)],
        invalid[:4],
    ):
        for item in group:
            key = (str(item.get("code_signature", "")), str(item.get("config_signature", "")))
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= max(elite_size * 5, 18):
                return selected
    return selected


def normalize_code_for_signature(code: str) -> str:
    """提取策略代码的结构签名文本，用于发现同质化候选。"""

    lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(('"""', "'''")):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def code_signature(code: str) -> str:
    """生成稳定的候选代码签名。"""

    normalized = normalize_code_for_signature(code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def config_signature(summary: dict[str, Any]) -> str:
    """生成核心配置签名，辅助发现同构策略。"""

    metrics = summary.get("validation_metrics", {})
    config = summary.get("config", {})
    payload = {
        "weight": metrics.get("completed_weight_within_horizon"),
        "setup": metrics.get("setup_count_positive"),
        "key_config": summarize_metrics(summary).get("key_config", {}),
    }
    if isinstance(config.get("force_path"), dict):
        payload["force_path_count"] = config["force_path"].get("count", len(config["force_path"]))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def build_strategy_api_spec() -> str:
    """描述 LLM 可编辑策略模块的接口。"""

    return textwrap.dedent(
        """
        你只能生成一个 Python 策略模块，不允许生成完整求解器。

        固定内核已经提供并禁止你修改：
        - 输入 JSON 解析、路径实例构建；
        - 候选动作生成和硬约束过滤；
        - qtime 最小/最大间隔、维修窗口、前后序、设备能力、跨厂转运、setup；
        - 有限组批容量、family、批时长、批组机器占用；
        - 两阶段调度主循环和完整解写出；
        - 人工编写的有限组批校验器。

        你的策略模块只能定义以下函数：

        ```python
        def describe_strategy() -> str:
            return "short description"

        def describe_rule_changes() -> dict:
            # 必填。用于说明本候选相对亲本/基线到底新增、删除、保留了哪些规则。
            return {
                "added_rules": ["具体新增规则；如果没有新增规则，写空列表并说明原因"],
                "removed_rules": ["具体删除或禁用的规则；例如让某个 score_* 返回 None"],
                "kept_rules": ["明确保留的有效规则片段"],
                "changed_parameters": {"参数名": "变化方向和理由"},
                "rule_suggestions": ["本轮未实施但下一轮值得尝试的规则新增、删减或消融建议"],
                "no_rule_change_reason": "若本轮只调参数，说明为什么暂不改规则",
                "expected_effect": "预期对产量、setup、完整性的影响",
                "risk": "可能导致退化的风险",
            }

        def get_config(base_config: dict, case_context: dict) -> dict:
            # 返回覆盖后的配置。只能修改启发式参数，不能读写文件。
            return base_config

        def score_path(features: dict) -> float | None:
            # 路径选择阶段：在输入 JSON 已有 process_path 中排序。
            # 返回 None 表示该任务使用固定内核默认路径规则。
            return None

        def score_phase1(features: dict) -> float:
            # 第一阶段：在固定内核给出的可行动作之间排序，目标是 horizon 内产量。
            return 0.0

        def score_phase2(features: dict) -> float:
            # 第二阶段：在固定内核给出的可行动作之间排序，目标是补齐完整合法解。
            return 0.0

        # 以下是更高层的可选“算子接口”。它们只会收到固定内核已经判定可行的
        # 候选动作，不能新增动作、机器或路径；若返回 None、越界或报错，内核会
        # 自动回退到默认窗口和评分函数。

        def select_phase1_candidates(candidates: list[dict], context: dict) -> list[int] | None:
            # 第一阶段候选池重构算子：返回 candidates 中要保留比较的局部下标。
            # 可用于扩大/缩小窗口、优先闭环任务、筛除低密度高 setup 候选等。
            return None

        def select_phase2_candidates(candidates: list[dict], context: dict) -> list[int] | None:
            # 第二阶段候选池重构算子：可用于优先已启动任务、保护 qtime 紧迫任务、
            # 或在低 setup 同族任务和尾部补齐任务之间做分层。
            return None

        def choose_phase1_action(candidates: list[dict], context: dict) -> int | None:
            # 第一阶段动作选择算子：直接返回 candidates 的局部下标。
            # 返回 None 表示继续由 score_phase1/default scoring 选择。
            return None

        def choose_phase2_action(candidates: list[dict], context: dict) -> int | None:
            # 第二阶段动作选择算子：直接返回 candidates 的局部下标。
            return None

        def score_batch_extra(anchor: dict, extra: dict, context: dict) -> float | None:
            # 有限组批成员扩展算子：当 anchor 已选中时，决定哪些同族额外任务
            # 更适合加入同一批。返回 None 表示使用默认组批排序。
            return None
        ```

        可覆盖配置键包括：
        - lookahead, start_guard
        - score_weight, score_density, score_started, score_family, score_zero_setup,
          score_setup_fixed, score_setup_per, score_est_final_per
        - phase2_started, phase2_density, phase2_family, phase2_zero_setup,
          phase2_setup_fixed, phase2_setup_per, phase2_finish_per,
          phase2_allow_unstarted
        - finite_batch_capacity, batch_group_wait, batch_group_mixed_time,
          batch_group_any_time
        - path_nonbatch_mult, path_batch_weight, path_wait_weight,
          path_machine_penalty
        - task_bonus, defer_task
        - operator_enable_hooks, operator_phase1_window, operator_phase2_window,
          operator_max_candidates

        score_path(features) 的主要字段：
        - task_id, path_id, task_weight, task_priority
        - delivery_time, earliest_available_time
        - process_count, nonbatch_count, batch_count
        - nonbatch_time, batch_time, min_wait, max_wait_count
        - machine_count, min_option_count, avg_option_count, estimated_path_time

        score_phase1/score_phase2(features) 以及候选池算子中单个 candidate 的主要字段：
        - candidate_index, phase, task_id, process_seq, process_index, process_count, path_id
        - machine_id, is_batch
        - task_weight, task_priority, remaining_nonbatch_time, density, progress
        - start, finish, estimated_final, can_finish_within_horizon, slack_to_horizon
        - horizon, current_time
        - start_delay_from_min, setup_time, same_family, zero_setup, started
        - option_priority, upper_bound, has_qtime_bound, q_slack, q_slack_raw

        候选池算子的 context 主要字段：
        - phase, candidate_count, started_count, batch_count
        - min_start, lookahead, horizon, current_time

        重要原则：
        - 不满足硬约束的动作根本不会传给你；你只负责在可行动作之间排序。
        - 路径选择也只能在输入 JSON 已有路径中排序；不能编造路径或机器。
        - 候选池算子和动作选择算子只能返回传入 candidates 的下标；不能返回 task_id、
          machine_id 或自造工序。越界、空列表、报错都会回退到默认逻辑。
        - 只有当 `get_config()` 设置 `operator_enable_hooks=True` 时，候选池算子、
          动作选择算子和组批扩展算子才会被调用；否则内核保持旧版评分函数路径。
        - “算子自进化”优先尝试候选池分层、直接动作选择、组批成员排序和阶段窗口
          调整，而不是只继续改 score_phase1/score_phase2 的线性权重。
        - 第一阶段偏产量：高权重、能闭环入窗、密度高、setup 少、同族连续优先。
        - 第二阶段偏完整：已启动任务、qtime 紧、尾部紧凑、setup 少优先。
        - 如果不确定自定义打分是否优于基线，`score_phase1/score_phase2` 可以返回
          None，让固定内核使用内置成熟评分；此时只通过 `get_config` 做小幅参数试探。
        - 每个候选必须有真实的规则审计：`describe_rule_changes()` 不能只写空话。
          允许纯参数微调，因为很多有效候选来自权重和阈值；但如果本轮只改参数，
          必须在 changed_parameters 中说明实质变化，并在 rule_suggestions 中提出
          下一轮可尝试的规则新增、删减或消融建议，同时说明 no_rule_change_reason。
        - 规则审计只写实验摘要，不要写成长篇报告；每个列表建议 1-3 条，每条
          尽量短。代码长度本身不会导致拒绝，但过长会降低后续交叉和复盘质量。
        - 规则审计必须自洽：同一条规则不能同时出现在 removed_rules 和
          added_rules/kept_rules 中。例如不能一边说删除 `score_phase2` 进度奖励，
          一边又在 kept_rules 中保留它。若发现冲突，先修改你的规则或审计说明。
        - “删除规则”可以表现为让某个 score_* 返回 None，或从自定义打分中去掉某个
          奖励/惩罚项；必须说明删除原因，例如产量下降、setup 反弹或与默认内核冲突。
        - 不要把 `upper_bound` 当作正向密度或正向奖励；很多无上界约束会接近 INF，
          会把排序完全冲坏。密度应使用 `task_weight / remaining_nonbatch_time`。
        - `q_slack` 始终是可比较数值；无 qtime 上界时会取很大的 INF 近似值。
          若要判断是否真的存在 qtime 约束，请使用 `has_qtime_bound` 或
          `q_slack_raw is not None`。
        - 对测试算例1，默认不要把 `batch_group_any_time` 改成 True；它会显著增加
          过度合批和尾部补齐风险。除非上一轮报告明确证明它改善完整合法指标。
        - 不要一次大幅改动十几个权重。优先围绕当前最佳合法基线做 1-3 个小改动，
          并配合一个可解释的规则新增、删除或消融动作。
        - 不允许 import os/sys/pathlib/subprocess/requests/urllib，不允许读写文件、
          调用 shell、调用校验器或导入项目求解器。
        - 不允许读取旧输出目录、旧解文件、旧自动调参日志或旧参数记录；经验只能来自
          本框架传入的上一轮摘要和校验反馈。
        - 输出必须只有一个 Python 代码块或纯 Python 源码。
        """
    ).strip()


def build_analyzer_prompt(
    *,
    previous_code: str,
    previous_summary: dict[str, Any],
    population: str,
    experience: str,
) -> str:
    """让 Analyzer 只分析策略问题，不写代码。"""

    return textwrap.dedent(
        f"""
        你是 FJSP 策略模块 Analyzer。固定内核负责所有硬约束，你只分析策略模块
        为什么产量/setup 不好，或者为什么被静态门禁拒绝。

        请输出 Markdown，包含：
        1. 主要问题；
        2. 候选中的好片段：哪些参数、score_path、score_phase1、score_phase2、
           候选池算子、动作选择算子或组批成员算子即使总分不佳也值得保留；
        3. 候选中的坏片段：哪些规则导致产量下降、setup 上升或完整性风险；
        4. 待变异片段：哪些阈值或权重应小步调整；
        5. 可删除规则：哪些自定义规则应被禁用、消融或回退到固定内核默认评分；
        6. 可新增规则/算子：只基于文档和候选特征，提出新的候选池分层、
           动作选择、组批扩展或排序因子组合方式；
        7. 下一轮策略修改建议；
        8. 禁止事项；
        9. 预期指标变化。

        ## 上一轮策略代码
        ```python
        {previous_code[-18000:]}
        ```

        ## 上一轮摘要
        ```json
        {json.dumps(previous_summary, ensure_ascii=False, indent=2)[-16000:]}
        ```

        ## 当前候选种群
        {population[-22000:]}

        ## 历史经验
        ```text
        {experience[-10000:]}
        ```
        """
    ).strip()


def build_generation_prompt(
    *,
    docs: list[tuple[Path, str]],
    mode: str,
    previous_code: str,
    previous_summary: dict[str, Any],
    best_summary: dict[str, Any],
    population: str,
    diversity_context: str,
    analyzer_advice: str,
    experience: str,
    target_weight: float,
    target_setup: int,
) -> str:
    """构造策略生成 prompt。"""

    doc_blocks = "\n\n".join(f"## 文档：{path.name}\n\n{text}" for path, text in docs)
    return textwrap.dedent(
        f"""
        你要生成 `deepseek_strategy.py`，它将在固定约束内核中运行。
        不要生成完整 solver，不要解析 JSON，不要写解文件。

        ## 目标
        1. 固定内核必须输出完整合法解。
        2. 本轮不是单纯追高产量，而是寻找 completed_weight 在 18500 附近且
           setup_count_positive 靠近 {target_setup} 的低切换策略。
        3. 框架会把完整合法性、产量、setup 和重复候选反馈给你；具体采用什么
           规则、参数、交叉或删减动作，由你根据反馈自行判断。

        {build_strategy_api_spec()}

        ## 本候选探索角色
        {mode}

        角色解释：
        - candidate_pool_operator：必须实现 select_phase1_candidates 或
          select_phase2_candidates 中至少一个候选池重构算子，并说明它如何改变
          “哪些可行动作进入比较”；必须在 get_config 中设置 operator_enable_hooks=True。
        - direct_action_operator：必须实现 choose_phase1_action 或 choose_phase2_action
          中至少一个直接动作选择算子，并说明它与线性评分函数的区别；必须打开
          operator_enable_hooks。
        - batch_group_operator：必须实现 score_batch_extra，改变有限组批时额外
          同族任务加入批组的优先级，同时保持 batch_group_any_time 默认为 False；
          必须打开 operator_enable_hooks。
        - phase_window_operator：必须通过 get_config 调整 operator_phase1_window 或
          operator_phase2_window，并配合候选池算子控制候选窗口，不要只改普通权重；
          必须打开 operator_enable_hooks。
        - new_dispatch_rule：必须在 score_phase1 中提出一个新的可解释派工规则，
          同时删除或禁用至少一个历史上无效的派工因子。
        - new_completion_rule：必须在 score_phase2 中提出一个新的可解释补全规则，
          同时删除或禁用至少一个历史上无效的补全因子。
        - rule_prune_ablation：以删规则/消融为主，必须让至少一个 score_* 回退到
          None、删除一个奖励/惩罚项，或禁用一个候选池/动作选择算子。
        - operator_crossover_prune：从精英候选中吸收好片段，同时删除坏片段并做
          小幅变异；优先交叉候选池算子、动作选择算子和参数窗口。
        - guarded_parameter_mutation：主要调 get_config，但仍必须在
          describe_rule_changes() 中说明保留了哪些规则、参数为什么这样改，以及
          下一轮可尝试的规则建议；允许本候选只改参数。

        ## 候选差异性硬要求
        同一轮候选之间不能同质化。你生成的策略必须满足：
        - 策略思路和上一候选不同，不能只改 describe_strategy 文案。
        - 必须实现 `describe_rule_changes()`，清楚列出 added_rules、removed_rules、
          kept_rules、changed_parameters、rule_suggestions、expected_effect、risk。
        - 允许本候选只改参数；但 changed_parameters 不能空，并且必须在
          rule_suggestions 里提出 1-3 个下一轮值得尝试的规则新增、删减或消融建议。
        - 如果本候选新增或删除规则，要说明它如何改变候选动作排序；如果本候选
          不改规则，要说明 no_rule_change_reason。
        - `describe_rule_changes()` 是简短审计，不是报告正文；避免大段注释和长篇
          docstring，把篇幅优先用于可执行规则、参数和阈值。
        - 至少一个函数的实质逻辑不同：get_config、score_path、score_phase1、
          score_phase2、select_phase*_candidates、choose_phase*_action、
          score_batch_extra 中至少一个要有新的规则结构或新的阈值组合。
        - 至少一个关键参数或阈值与本轮已尝试候选不同，且差异要足以改变调度排序。
        - 如果沿用精英候选的好片段，必须同时引入一个明确的交叉、变异或删减动作。
        - 这里的“片段”只指策略经验、评分因子、函数逻辑或参数组合，不是排程解
          的局部结构；禁止读取、复制或拼接任何旧解文件中的工序安排。
        - 禁止连续输出与本轮已尝试候选指标和配置高度相同的策略。
        - 如果上一候选与历史候选指标高度相同，本候选必须自行判断原因，并改变
          足以影响排序的规则或参数。
        - 如果本轮已尝试候选被标记为 duplicate_config 或 poor_rule_change_audit，
          你必须把它当作失败样本：不要继续沿用同一组核心参数、同一套规则组合，
          也不要重复 added/removed/kept 自相矛盾的审计。
        - 不要为了接近 setup 目标而接受明显低产平台。若产量低于 {target_weight - 300:.0f}，
          必须说明如何把产量拉回目标带，而不是只继续压低 setup。

        本轮已尝试候选签名和摘要：
        ```json
        {diversity_context[-12000:] if diversity_context else "[]"}
        ```

        ## Analyzer 建议
        {analyzer_advice or "暂无。"}

        ## 上一轮策略代码
        ```python
        {previous_code[-18000:] if previous_code else "暂无。"}
        ```

        ## 上一轮运行摘要
        ```json
        {json.dumps(previous_summary, ensure_ascii=False, indent=2)[-16000:] if previous_summary else "{}"}
        ```

        ## 当前最佳合法基线
        ```json
        {json.dumps(best_summary, ensure_ascii=False, indent=2)[-16000:] if best_summary else "{}"}
        ```

        你生成的新策略必须尝试超过当前最佳合法基线；如果上一轮比基线差，
        请优先回退到基线参数附近，只做小幅结构化改动。
        如果上一轮出现 `fully_scheduled_tasks < total_tasks`，说明策略破坏了二阶段补齐，
        下一轮应让 score_phase1/score_phase2 返回 None 或接近基线公式，并撤销激进组批设置。

        ## 当前候选种群与可交叉亲本
        {population[-26000:]}

        你可以做的“规则演化”包括：
        - 产生新规则/算子：围绕文档约束、候选特征和历史指标，自主提出新的
          候选池筛选、动作选择、组批扩展或排序规则。
        - 交叉规则/算子：组合两个精英候选中各自有效的 get_config、候选池算子、
          动作选择算子、score_path、score_phase2。
        - 变异规则：只改变 1-3 个权重或阈值，不要大面积重写。
        - 删除规则/算子：如果某条规则使产量下降、任务不完整或 setup 异常，可让对应
          score_* 或 select/choose hook 返回 None，回退到固定内核成熟逻辑。
        - 消融验证：保留参数不变，只删除一个疑似无效规则，验证它是否真的有贡献。
        - 以上操作都只发生在策略模块层面，不允许使用旧解的局部排程结构。

        不要照抄上述角色名当策略。你需要根据文档、接口和候选种群反馈自行判断：
        - 哪些规则片段应保留；
        - 哪些规则片段应删除；
        - 哪些规则片段应交叉；
        - 哪些参数或阈值应变异。

        ## 历史经验
        ```text
        {experience[-12000:] if experience else "暂无。"}
        ```

        {doc_blocks}
        """
    ).strip()


def run_kernel(
    *,
    root: Path,
    input_path: Path,
    track: str,
    strategy_path: Path,
    solution_path: Path,
    summary_path: Path,
    timeout: int,
) -> tuple[int | None, str, str, dict[str, Any]]:
    """运行固定内核并读取 summary。"""

    cmd = [
        sys.executable,
        "scripts/strategy_kernel_solver.py",
        "--root",
        str(root),
        "--input",
        str(input_path),
        "--track",
        track,
        "--strategy",
        str(strategy_path),
        "--output",
        str(solution_path),
        "--summary",
        str(summary_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr, read_json(summary_path)
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout or "", exc.stderr or "kernel timeout", {}


def experience_entry(
    *,
    iteration: int,
    summary: dict[str, Any],
    score: float,
    stdout: str,
    stderr: str,
) -> str:
    """生成一条经验记录。"""

    metrics = summary.get("validation_metrics", {})
    return textwrap.dedent(
        f"""
        ### Iteration {iteration}

        - strategy_name: {summary.get("strategy_name", "NA")}
        - error_count: {summary.get("error_count", "NA")}
        - completed_weight_within_horizon: {metrics.get("completed_weight_within_horizon")}
        - setup_count_positive: {metrics.get("setup_count_positive")}
        - fully_scheduled_tasks: {metrics.get("fully_scheduled_tasks")}/{metrics.get("total_tasks")}
        - score: {score}
        - rule_changes:
        ```json
        {json.dumps(summary.get("rule_changes", {}), ensure_ascii=False, indent=2)}
        ```
        - sample_errors: {summary.get("sample_errors", [])[:5]}

        stdout tail:
        ```text
        {stdout[-2500:]}
        ```

        stderr tail:
        ```text
        {stderr[-2500:]}
        ```
        """
    ).strip()


def record_best(
    *,
    output_dir: Path,
    score: float,
    candidate_path: Path,
    solution_path: Path,
    summary: dict[str, Any],
) -> None:
    """保存当前最佳合法策略和解。"""

    shutil.copyfile(candidate_path, output_dir / "best_deepseek_strategy.py")
    if solution_path.exists():
        shutil.copyfile(solution_path, output_dir / "best_solution.json")
    save_json(
        output_dir / "best_summary.json",
        {
            "score": score,
            "strategy": str(candidate_path),
            "solution": str(solution_path),
            "summary": summary,
        },
    )


def extract_number(path: Path) -> int:
    """从报告/提示文件名中提取最后一个数字；没有数字时返回 -1。"""

    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def restore_population(output_dir: Path) -> list[dict[str, Any]]:
    """恢复本框架上一轮写出的候选种群。"""

    data = read_json(output_dir / "population.json")
    raw_population = data.get("population", [])
    return raw_population if isinstance(raw_population, list) else []


def restore_best(output_dir: Path) -> tuple[float, dict[str, Any], Path | None, Path | None]:
    """恢复本框架上一轮记录的 best。"""

    data = read_json(output_dir / "best_summary.json")
    if not data:
        return float("-inf"), {}, None, None
    score = safe_number(data.get("score"), float("-inf"))
    strategy = Path(str(data["strategy"])) if data.get("strategy") else None
    solution = Path(str(data["solution"])) if data.get("solution") else None
    summary = data.get("summary", {})
    return score, summary if isinstance(summary, dict) else {}, strategy, solution


def refresh_population_scores(
    population: list[dict[str, Any]],
    setup_penalty: float,
    *,
    target_weight: float = 0.0,
    target_setup: int = 0,
) -> list[dict[str, Any]]:
    """按本次运行的目标权重重新计算历史种群得分。

    同一输出目录可能被不同目标继续使用，例如从“高产优先”切换到
    “185xx/500 左右”。历史 population.json 中的 score 只代表当时的目标
    函数；续跑时必须用当前 setup_penalty 重新排序，避免旧目标下的 best
    阻止新目标候选被记录。
    """

    refreshed: list[dict[str, Any]] = []
    for item in population:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            continue
        updated = dict(item)
        updated["score"] = score_summary(
            summary,
            setup_penalty,
            target_weight=target_weight,
            target_setup=target_setup,
        )
        refreshed.append(updated)
    refreshed.sort(key=lambda item: safe_number(item.get("score"), float("-inf")), reverse=True)
    return refreshed


def best_from_population(
    population: list[dict[str, Any]],
) -> tuple[float, dict[str, Any], Path | None, Path | None]:
    """从已按当前目标打分的种群中选出合法完整 best。"""

    for item in population:
        summary = item.get("summary", {})
        if not isinstance(summary, dict) or not is_complete_legal(summary):
            continue
        score = safe_number(item.get("score"), float("-inf"))
        candidate = item.get("candidate_path") or summary.get("candidate") or summary.get("strategy")
        solution = item.get("solution_path") or summary.get("solution") or summary.get("output")
        candidate_path = Path(str(candidate)) if candidate else None
        solution_path = Path(str(solution)) if solution else None
        return score, summary, candidate_path, solution_path
    return float("-inf"), {}, None, None


def safe_number(value: Any, fallback: float) -> float:
    """安全转换浮点数。"""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if not (math.isnan(result) or math.isinf(result)) else fallback


def next_report_index(output_dir: Path) -> int:
    """从现有 reports 中推断下一条运行编号。"""

    reports_dir = output_dir / "reports"
    if not reports_dir.exists():
        return 0
    max_idx = -1
    for path in reports_dir.glob("framework_*_iter*.json"):
        max_idx = max(max_idx, extract_number(path))
    return max_idx + 1


def next_round_index(output_dir: Path) -> int:
    """从现有 prompts 中推断下一轮编号，避免续跑覆盖文件。"""

    prompts_dir = output_dir / "prompts"
    if not prompts_dir.exists():
        return 0
    max_round = -1
    for path in prompts_dir.glob("round_*_cand_*_prompt.md"):
        match = re.search(r"round_(\d+)_cand_", path.name)
        if match:
            max_round = max(max_round, int(match.group(1)))
    return max_round + 1


def latest_summary(output_dir: Path) -> tuple[dict[str, Any], str]:
    """读取最近一次候选摘要和对应代码，用作下一轮 Analyzer 输入。"""

    reports_dir = output_dir / "reports"
    if not reports_dir.exists():
        return {}, ""
    candidates = sorted(
        reports_dir.glob("framework_summary_iter*.json"),
        key=lambda path: (extract_number(path), path.stat().st_mtime_ns),
    )
    if not candidates:
        return {}, ""
    summary = read_json(candidates[-1])
    code = ""
    candidate_path = Path(str(summary.get("candidate", "")))
    if candidate_path.exists():
        code = candidate_path.read_text(encoding="utf-8", errors="replace")
    return summary, code


def main() -> int:
    """策略自迭代主流程。"""

    args = parse_args()
    root = args.root.resolve()
    input_path = resolve_path(root, args.input)
    output_dir = resolve_path(root, args.output_dir)
    active_strategy_path = resolve_path(root, args.strategy_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        print(f"[error] {args.api_key_env} is not set", file=sys.stderr)
        return 2

    docs = load_documents(root, args.doc or DEFAULT_DOCS)
    seed_strategy = root / "strategies" / "deepseek_strategy.py"
    previous_code = seed_strategy.read_text(encoding="utf-8", errors="replace")
    previous_summary: dict[str, Any] = {}
    analyzer_advice = ""
    best_score = float("-inf")
    best_summary: dict[str, Any] = {}
    best_candidate_path: Path | None = None
    best_solution_path: Path | None = None
    population: list[dict[str, Any]] = []
    seen_code_signatures: set[str] = set()
    seen_config_signatures: set[str] = set()
    experience_path = output_dir / "experience_log.md"
    experience = (
        experience_path.read_text(encoding="utf-8", errors="replace")
        if experience_path.exists()
        else ""
    )
    restored_population = restore_population(output_dir)
    if restored_population:
        population = select_population(
            refresh_population_scores(
                restored_population,
                args.setup_penalty,
                target_weight=args.target_weight,
                target_setup=args.target_setup,
            ),
            elite_size=args.elite_size,
            target_weight=args.target_weight,
            target_setup=args.target_setup,
        )
        for item in population:
            code_sig = item.get("code_signature")
            cfg_sig = item.get("config_signature")
            if code_sig:
                seen_code_signatures.add(str(code_sig))
            if cfg_sig:
                seen_config_signatures.add(str(cfg_sig))
        best_score, best_summary, best_candidate_path, best_solution_path = best_from_population(population)
        if best_candidate_path is None:
            best_score, best_summary, best_candidate_path, best_solution_path = restore_best(output_dir)
        latest_prev_summary, latest_prev_code = latest_summary(output_dir)
        if latest_prev_summary:
            previous_summary = latest_prev_summary
        elif best_summary:
            previous_summary = best_summary
        if latest_prev_code:
            previous_code = latest_prev_code
        elif best_candidate_path and best_candidate_path.exists():
            previous_code = best_candidate_path.read_text(encoding="utf-8", errors="replace")
        print(
            "[strategy-framework] restored population: "
            f"size={len(population)}, best_score={best_score}",
            flush=True,
        )

    if not restored_population and not args.no_seed_baseline and not args.dry_run:
        baseline_solution = output_dir / "baseline" / "solution.json"
        baseline_summary_path = output_dir / "baseline" / "summary.json"
        exit_code, stdout, stderr, baseline_summary = run_kernel(
            root=root,
            input_path=input_path,
            track=args.track,
            strategy_path=seed_strategy,
            solution_path=baseline_solution,
            summary_path=baseline_summary_path,
            timeout=args.solver_timeout,
        )
        baseline_summary["kernel_exit_code"] = exit_code
        baseline_summary["candidate"] = str(seed_strategy)
        baseline_summary["solution"] = str(baseline_solution)
        baseline_score = score_summary(
            baseline_summary,
            args.setup_penalty,
            target_weight=args.target_weight,
            target_setup=args.target_setup,
        )
        save_json(output_dir / "baseline" / "framework_summary.json", baseline_summary)
        entry = experience_entry(
            iteration=-1,
            summary=baseline_summary,
            score=baseline_score,
            stdout=stdout,
            stderr=stderr,
        )
        append_text(experience_path, entry)
        experience += "\n\n" + entry
        previous_summary = baseline_summary
        best_summary = baseline_summary
        if is_complete_legal(baseline_summary):
            best_score = baseline_score
            best_candidate_path = seed_strategy
            best_solution_path = baseline_solution
            baseline_code_sig = code_signature(previous_code)
            baseline_config_sig = config_signature(baseline_summary)
            seen_code_signatures.add(baseline_code_sig)
            seen_config_signatures.add(baseline_config_sig)
            population.append(
                {
                    "score": baseline_score,
                    "candidate_path": str(seed_strategy),
                    "solution_path": str(baseline_solution),
                    "summary": baseline_summary,
                    "code_signature": baseline_code_sig,
                    "config_signature": baseline_config_sig,
                }
            )
            record_best(
                output_dir=output_dir,
                score=best_score,
                candidate_path=seed_strategy,
                solution_path=baseline_solution,
                summary=baseline_summary,
            )
        print(
            "[strategy-framework] seeded baseline: "
            f"score={baseline_score}, "
            f"metrics={baseline_summary.get('validation_metrics', {})}",
            flush=True,
        )

    run_index = next_report_index(output_dir)
    round_offset = next_round_index(output_dir)
    for iteration in range(args.max_iters):
        actual_round = round_offset + iteration
        current_population = population_brief(population, elite_size=args.elite_size)
        round_diversity_records: list[dict[str, Any]] = []
        if previous_summary:
            analyzer_prompt = build_analyzer_prompt(
                previous_code=previous_code,
                previous_summary=previous_summary,
                population=current_population,
                experience=experience,
            )
            analyzer_prompt_path = output_dir / "analysis" / f"round_{actual_round:02d}_prompt.md"
            analyzer_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            analyzer_prompt_path.write_text(analyzer_prompt, encoding="utf-8")
            if not args.dry_run:
                analyzer_advice = call_deepseek(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=analyzer_prompt,
                    temperature=min(args.temperature, 0.15),
                    max_tokens=min(args.max_tokens, 4096),
                    system_prompt="你是组合优化策略分析员，只输出诊断建议，不输出代码。",
                    wire_api=args.wire_api,
                    timeout=args.llm_timeout,
                )
                (output_dir / "analysis" / f"round_{actual_round:02d}_advice.md").write_text(
                    analyzer_advice, encoding="utf-8"
                )

        for candidate_index in range(max(1, args.candidates_per_iter)):
            prefix = f"round_{actual_round:02d}_cand_{candidate_index:02d}"
            mode = candidate_mode(iteration, candidate_index)
            prompt = build_generation_prompt(
                docs=docs,
                mode=mode,
                previous_code=previous_code,
                previous_summary=previous_summary,
                best_summary=best_summary,
                population=current_population,
                diversity_context=json.dumps(round_diversity_records, ensure_ascii=False, indent=2),
                analyzer_advice=analyzer_advice,
                experience=experience,
                target_weight=args.target_weight,
                target_setup=args.target_setup,
            )
            prompt_path = output_dir / "prompts" / f"{prefix}_prompt.md"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
            print(f"[strategy-framework] prompt saved: {prompt_path}", flush=True)
            if args.dry_run:
                return 0

            response = call_deepseek(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                wire_api=args.wire_api,
                timeout=args.llm_timeout,
            )
            response_path = output_dir / "responses" / f"{prefix}_response.txt"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(response, encoding="utf-8")

            code = extract_python_code(response)
            code_sig = code_signature(code)
            candidate_path = output_dir / "candidates" / f"deepseek_strategy_{prefix}.py"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(code, encoding="utf-8")

            issues = check_strategy_safety(code)
            if issues:
                previous_summary = {
                    "status": "rejected_by_strategy_safety",
                    "issues": issues,
                    "mode": mode,
                    "candidate": str(candidate_path),
                }
                previous_code = code
                save_json(output_dir / "reports" / f"framework_reject_iter{run_index:02d}.json", previous_summary)
                print(json.dumps(previous_summary, ensure_ascii=False, indent=2), flush=True)
                run_index += 1
                continue

            if code_sig in seen_code_signatures:
                previous_summary = {
                    "status": "rejected_by_duplicate_code",
                    "issues": [
                        "candidate code is structurally too similar to an earlier candidate; generate a substantively different rule or parameter mutation"
                    ],
                    "mode": mode,
                    "candidate": str(candidate_path),
                    "code_signature": code_sig,
                    "round_diversity_records": round_diversity_records,
                }
                previous_code = code
                save_json(
                    output_dir / "reports" / f"framework_duplicate_code_iter{run_index:02d}.json",
                    previous_summary,
                )
                print(json.dumps(previous_summary, ensure_ascii=False, indent=2), flush=True)
                round_diversity_records.append(
                    {
                        "candidate": str(candidate_path),
                        "mode": mode,
                        "status": "duplicate_code",
                        "code_signature": code_sig,
                    }
                )
                run_index += 1
                continue

            shutil.copyfile(candidate_path, active_strategy_path)
            solution_path = output_dir / "solutions" / f"solution_iter{run_index:02d}.json"
            summary_path = output_dir / "summaries" / f"summary_iter{run_index:02d}.json"
            exit_code, stdout, stderr, summary = run_kernel(
                root=root,
                input_path=input_path,
                track=args.track,
                strategy_path=candidate_path,
                solution_path=solution_path,
                summary_path=summary_path,
                timeout=args.solver_timeout,
            )
            summary["kernel_exit_code"] = exit_code
            summary["candidate"] = str(candidate_path)
            summary["solution"] = str(solution_path)
            summary["mode"] = mode
            summary["code_signature"] = code_sig
            cfg_sig = config_signature(summary)
            summary["config_signature"] = cfg_sig
            rule_change_issues = audit_rule_changes(summary.get("rule_changes"))
            if rule_change_issues:
                summary["rule_change_quality_warning"] = rule_change_issues
            duplicate_config = cfg_sig in seen_config_signatures
            if duplicate_config:
                summary["diversity_warning"] = (
                    "core metrics and key config are highly similar to an earlier candidate; "
                    "this candidate will not enter population; next candidates should use "
                    "a substantially different rule family"
                )
            save_json(summary_path, summary)
            score = score_summary(
                summary,
                args.setup_penalty,
                target_weight=args.target_weight,
                target_setup=args.target_setup,
            )
            save_json(output_dir / "reports" / f"framework_summary_iter{run_index:02d}.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

            entry = experience_entry(
                iteration=run_index,
                summary=summary,
                score=score,
                stdout=stdout,
                stderr=stderr,
            )
            append_text(experience_path, entry)
            experience += "\n\n" + entry

            previous_code = code
            previous_summary = summary
            seen_code_signatures.add(code_sig)
            seen_config_signatures.add(cfg_sig)
            if rule_change_issues:
                round_diversity_records.append(
                    {
                        "candidate": str(candidate_path),
                        "mode": mode,
                        "status": "poor_rule_change_audit",
                        "score": score,
                        "code_signature": code_sig,
                        "config_signature": cfg_sig,
                        "rule_change_issues": rule_change_issues,
                        "metrics": summarize_metrics(summary),
                    }
                )
                run_index += 1
                time.sleep(1)
                continue
            if duplicate_config:
                round_diversity_records.append(
                    {
                        "candidate": str(candidate_path),
                        "mode": mode,
                        "status": "duplicate_config",
                        "score": score,
                        "code_signature": code_sig,
                        "config_signature": cfg_sig,
                        "metrics": summarize_metrics(summary),
                    }
                )
                run_index += 1
                time.sleep(1)
                continue
            round_diversity_records.append(
                {
                    "candidate": str(candidate_path),
                    "mode": mode,
                    "status": "evaluated",
                    "score": score,
                    "code_signature": code_sig,
                    "config_signature": cfg_sig,
                    "metrics": summarize_metrics(summary),
                }
            )
            population.append(
                {
                    "score": score,
                    "candidate_path": str(candidate_path),
                    "solution_path": str(solution_path),
                    "summary": summary,
                    "code_signature": code_sig,
                    "config_signature": cfg_sig,
                }
            )
            population = select_population(
                population,
                elite_size=args.elite_size,
                target_weight=args.target_weight,
                target_setup=args.target_setup,
            )
            save_json(output_dir / "population.json", {"population": population})
            if score > best_score:
                best_score = score
                best_summary = summary
                best_candidate_path = candidate_path
                best_solution_path = solution_path
                record_best(
                    output_dir=output_dir,
                    score=best_score,
                    candidate_path=candidate_path,
                    solution_path=solution_path,
                    summary=summary,
                )
                print("[strategy-framework] new best legal strategy recorded", flush=True)

            metrics = summary.get("validation_metrics", {})
            if (
                is_complete_legal(summary)
                and metric_float(metrics, "completed_weight_within_horizon") >= args.target_weight
                and metric_int(metrics, "setup_count_positive") <= args.target_setup
            ):
                print("[strategy-framework] target reached; stop early", flush=True)
                return 0

            run_index += 1
            time.sleep(1)

    final = {
        "best_score": best_score,
        "best_strategy": str(best_candidate_path) if best_candidate_path else None,
        "best_solution": str(best_solution_path) if best_solution_path else None,
        "best_summary": best_summary,
    }
    save_json(output_dir / "final_report.json", final)
    return 0 if best_candidate_path is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

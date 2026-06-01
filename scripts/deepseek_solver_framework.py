#!/usr/bin/env python3
"""DeepSeek 驱动的调度求解器自迭代框架原型。

本脚本不是直接求解 FJSP，而是负责把三份需求/技术文档、现有有限组批
校验口径和实验目标组织成 Prompt，调用 DeepSeek 生成 `deepseek_solver.py`，
随后运行生成的求解器，并用人工编写的 `validate_batch_solution.py` 作为
唯一合法性裁判。若求解器运行失败、输出非法或指标不达标，脚本会把日志
和校验反馈再次提交给 DeepSeek，要求其重写求解器，形成“生成-运行-校验-
反馈-再生成”的小闭环。

安全边界：
1. API Key 只从环境变量读取，不写入 prompt、日志、生成代码或输出文件。
2. 生成的 solver 不允许调用网络、删除文件或修改校验器。
3. 合法性只以人工校验器为准，LLM 的解释不能覆盖校验结果。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DOCS = (
    "问题描述文档.md",
    "输入输出结构.md",
    "rl1.md",
)

FORBIDDEN_CODE_PATTERNS = (
    "requests.",
    "urllib.",
    "http.client",
    "socket.",
    "openai",
    "anthropic",
    "subprocess.",
    "shutil.rmtree",
    "os.remove",
    ".unlink(",
    ".rmdir(",
    "git ",
    "subprocess.run",
    "subprocess.Popen",
)


@dataclass
class ValidationResult:
    """一次候选 solver 运行后的外部证据。"""

    iteration: int
    solver_exit_code: int | None
    validator_exit_code: int | None
    solution_path: Path
    report_path: Path
    metrics: dict[str, Any]
    error_count: int
    solver_stdout: str
    solver_stderr: str
    validator_stdout: str
    validator_stderr: str


@dataclass
class BestArtifact:
    """当前迭代过程中发现的最佳合法候选。"""

    iteration: int
    score: float
    completed_weight: float
    setup_count: int
    solver_candidate_path: Path
    solution_path: Path
    report_path: Path
    metrics: dict[str, Any]


@dataclass
class BestAttempt:
    """尚未合法时，用于防止迭代退化的最佳尝试记录。"""

    iteration: int
    attempt_key: tuple[float, int, float]
    fully_scheduled_tasks: int
    total_tasks: int
    error_count: int
    completed_weight: float
    setup_count: int
    solver_candidate_path: Path
    solution_path: Path
    report_path: Path
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """解析框架运行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Generate and iterate deepseek_solver.py with DeepSeek, then verify it "
            "with the finite-batch validator."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/data1/测试算例1.json"),
        help="有限组批实验使用的输入算例。",
    )
    parser.add_argument(
        "--solver-path",
        type=Path,
        default=Path("deepseek_solver.py"),
        help="DeepSeek 生成的求解器文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/deepseek_framework"),
        help="保存 prompt、响应、候选代码、解文件和校验报告的目录。",
    )
    parser.add_argument(
        "--doc",
        action="append",
        default=[],
        help="额外或替代文档路径；不传时使用问题描述、输入输出结构、rl1。",
    )
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
    parser.add_argument(
        "--prompt-mode",
        choices=("strict", "wrapper"),
        default="strict",
        help=(
            "strict=只给文档、校验器和JSON读写模板；"
            "wrapper=允许提示现有求解器接口（仅用于工程复用实验）。"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=180,
        help="单次 LLM HTTP 请求超时时间，单位秒。",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="最多进行多少轮生成/修复。",
    )
    parser.add_argument(
        "--candidates-per-iter",
        type=int,
        default=1,
        help="每轮让 LLM 生成多少个候选 solver；>1 时形成小种群并自动选择最佳合法候选。",
    )
    parser.add_argument(
        "--setup-penalty",
        type=float,
        default=0.2,
        help="比较合法候选时，每多一次正 setup 折算的产量惩罚。",
    )
    parser.add_argument(
        "--max-presets-per-solver",
        type=int,
        default=2,
        help="提示 LLM 在单个生成 solver 内最多评估多少组 preset。",
    )
    parser.add_argument(
        "--previous-code-chars",
        type=int,
        default=28000,
        help="续跑时放入 Coder prompt 的上一轮代码最大字符数。",
    )
    parser.add_argument(
        "--two-stage-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 Analyzer -> Coder 二阶段反思。Analyzer 先读校验报告并提出修复意见。",
    )
    parser.add_argument(
        "--skip-analyzer",
        action="store_true",
        help="续跑时跳过单独 Analyzer API 调用，但仍把上一轮代码和运行反馈交给 Coder。",
    )
    parser.add_argument(
        "--solver-timeout",
        type=int,
        default=420,
        help="单个生成 solver 的最长运行秒数，超时会作为反馈进入下一轮。",
    )
    parser.add_argument(
        "--validator-timeout",
        type=int,
        default=420,
        help="有限组批校验器的最长运行秒数。",
    )
    parser.add_argument(
        "--target-weight",
        type=float,
        default=18835.01,
        help="有限组批当前高产量参考目标。达到或超过该值时可提前停止。",
    )
    parser.add_argument(
        "--target-setup",
        type=int,
        default=856,
        help="在达到目标产量时希望 setup 不高于该参考值。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成首轮 prompt，不调用 DeepSeek，不写 solver。",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="只生成 solver，不运行求解和校验。",
    )
    parser.add_argument(
        "--allow-unsafe-code",
        action="store_true",
        help="关闭生成代码的危险模式检查。默认不建议使用。",
    )
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    """将命令行路径统一解析为绝对路径。"""

    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_text(path: Path, limit_chars: int | None = None) -> str:
    """读取文本文件，并在需要时截断，避免 prompt 失控。"""

    text = path.read_text(encoding="utf-8", errors="replace")
    if limit_chars is not None and len(text) > limit_chars:
        return text[:limit_chars] + "\n\n[文档过长，后续内容已由框架截断]\n"
    return text


def load_documents(root: Path, doc_args: Iterable[str]) -> list[tuple[Path, str]]:
    """加载本轮提供给 LLM 的需求/技术文档。"""

    doc_names = tuple(doc_args) if doc_args else DEFAULT_DOCS
    docs: list[tuple[Path, str]] = []
    for name in doc_names:
        path = resolve_path(root, Path(name))
        if not path.exists():
            raise FileNotFoundError(f"document not found: {path}")
        docs.append((path, read_text(path, limit_chars=28000)))
    return docs


def build_wrapper_context_summary() -> str:
    """给 LLM 的现有代码接口摘要。

    这里故意不要求 LLM 从零解析全部输入 JSON，而是允许它复用已经人工验证过
    的 parser、scheduler 和 finite-batch validator。这样首版框架更像“代码
    演化/启发式演化”，而不是把大量工程细节一次性交给模型硬猜。
    """

    return textwrap.dedent(
        """
        可复用的本地接口如下：

        1. `scripts/rl_relaxed_solver.py`
           - 必须使用 `pathlib.Path`，不要把字符串传给这些接口。
           - `build_instance(root: Path, input_path: Path, cache_path: Path, force: bool=False,
             path_nonbatch_mult: float=3.0, path_batch_weight: float=1.0,
             path_wait_weight: float=1.0, path_machine_penalties=None,
             force_path_map=None, current_time_override=None,
             zero_current_time=False, maintenance_shift: int=0)`
           - `SetupRowStore(setup_db: Path).ensure(input_path: Path, force=False)`
           - `RelaxedRLScheduler(instance, setup_store, ..., finite_batch_capacity=True, batch_group_wait=..., batch_group_mixed_time=..., batch_group_any_time=...)`
           - `scheduler.solve()` 返回 task_records
           - `scheduler.metrics()` 返回 completed_weight、setup_count 等指标
           - `dump_solution(instance, task_records, output_path)` 写出标准解 JSON

        2. `scripts/validate_batch_solution.py`
           - 命令行校验入口：
             `python scripts/validate_batch_solution.py --input <input.json> --solution <solution.json> --report <report.json> --max-errors 30`
           - 这是有限组批口径的唯一合法性裁判。
           - 生成 solver 不允许修改该文件，也不允许自行放宽约束。

        3. 当前有限组批参考结果
           - 测试算例1、horizon=24480、完整 1751 个任务。
           - 已知高产量参考点约为 completed_weight=18835.01，setup_count_positive=856。
           - 已知更平衡参考点约为 completed_weight=18808.59，setup_count_positive=659。
           - 目标优先级：先合法完整，再提高 completed_weight，再降低正 setup 次数。

        4. 生成的 `deepseek_solver.py` 必须满足
           - 位于项目根目录运行。
           - 支持参数：`--root`、`--input`、`--output`、`--rebuild-instance`、`--rebuild-setup`。
           - 必须输出包含所有任务的完整解。
           - 必须启用有限组批：`finite_batch_capacity=True`。
           - 可以在文件内部定义多个候选 preset，逐个求解、临时校验，最后保留合法且指标最好的解。
           - 不允许调用网络 API，不允许删除文件，不允许修改校验器。

        5. 最小可运行模板（必须遵守 Path 类型和参数名）
           ```python
           from pathlib import Path
           import argparse, json, shutil, sys

           def main():
               args = parse_args()
               root = Path(args.root).resolve()
               input_path = Path(args.input).resolve()
               output_path = Path(args.output).resolve()
               sys.path.insert(0, str(root / "scripts"))
               from rl_relaxed_solver import build_instance, SetupRowStore, RelaxedRLScheduler, dump_solution

               instance_cache = root / "cache" / "deepseek_instance.pkl"
               setup_db = root / "cache" / "deepseek_setup.sqlite"
               instance = build_instance(
                   root=root,
                   input_path=input_path,
                   cache_path=instance_cache,
                   force=args.rebuild_instance,
                   path_nonbatch_mult=3.0,
                   path_batch_weight=1.0,
                   path_wait_weight=1.0,
                   path_machine_penalties=None,
                   force_path_map=None,
                   current_time_override=None,
                   zero_current_time=False,
                   maintenance_shift=0,
               )
               setup_store = SetupRowStore(setup_db)
               setup_store.ensure(input_path, force=args.rebuild_setup)
               try:
                   scheduler = RelaxedRLScheduler(
                       instance,
                       setup_store,
                       lookahead=85,
                       start_guard=120,
                       finite_batch_capacity=True,
                       batch_group_wait=320,
                       batch_group_mixed_time=True,
                       batch_group_any_time=False,
                   )
                   records = scheduler.solve()
                   dump_solution(instance, records, output_path)
                   print(json.dumps(scheduler.metrics(), ensure_ascii=False, indent=2))
               finally:
                   setup_store.close()
           ```
        """
    ).strip()


def build_strict_context_summary() -> str:
    """严格实验口径：只暴露文档、校验器和极薄 JSON 读写模板。"""

    return textwrap.dedent(
        """
        本轮是严格文档驱动实验，生成的 `deepseek_solver.py` 必须独立实现调度逻辑。

        允许信息：
        1. 三份文档：问题描述、输入输出结构、rl1。
        2. 原始输入 JSON 文件，由 `--input` 指定。
        3. 有限组批校验器作为外部黑盒裁判，外层框架会调用：
           `python scripts/validate_batch_solution.py --input <input.json> --solution <solution.json> --report <report.json> --max-errors 40`
        4. 一个最小 JSON 读写模板。

        禁止信息与禁止行为：
        1. 不允许导入或调用项目中已经存在的求解、解析、调参或排程脚本。
        2. 不允许复用项目中已经实现好的调度器、缓存构建器、路径选择器或解文件写出器。
        3. 不允许读取已有输出解、历史 best 文件或调参结果文件。
        4. 不允许修改或导入校验器；校验器只由外层框架命令行调用。
        5. 不允许调用网络 API、shell 命令、git 命令或删除文件。

        最小 JSON 读写模板：
        ```python
        from pathlib import Path
        import argparse
        import json

        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--root", type=str, default=".")
            parser.add_argument("--input", type=str, required=True)
            parser.add_argument("--output", type=str, required=True)
            parser.add_argument("--rebuild-instance", action="store_true")
            parser.add_argument("--rebuild-setup", action="store_true")
            return parser.parse_args()

        def main():
            args = parse_args()
            input_path = Path(args.input).resolve()
            output_path = Path(args.output).resolve()
            with input_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            # TODO: 根据文档自行解析任务、工序、候选设备、释放时间、q-time、
            # setup、维修窗口、转运、有限组批容量等信息，并生成完整调度解。
            solution = {}

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(solution, fh, ensure_ascii=False, indent=2)

        if __name__ == "__main__":
            main()
        ```

        生成代码可以使用 Python 标准库中的 `argparse`、`json`、`pathlib`、
        `dataclasses`、`collections`、`heapq`、`math`、`random`、`datetime`。
        """
    ).strip()


def build_internal_two_stage_requirements() -> str:
    """生成 solver 内部应采用的两阶段启发式框架说明。"""

    return textwrap.dedent(
        """
        ## 生成 solver 内部必须体现的两阶段调度框架

        这里的“两阶段”指生成出来的 `deepseek_solver.py` 自己的排程逻辑，不是外层
        LLM 反思流程：

        ### 硬约束动作过滤（必须实现）
        - 调度器不能采用“先排程、最后再校验”的方式。每一次选择工序/设备/开始时间/批组
          之前，必须构造候选动作并调用类似 `is_feasible_action(action, state)` 的检查。
        - 只有通过硬约束检查的动作才能进入评分排序；不满足约束的动作必须直接丢弃，
          不能通过扣分、负奖励或后处理修补继续保留。
        - 硬约束检查至少覆盖：任务释放时间、前驱工序存在且完成、所选路径顺序、
          qtime 最小/最大间隔、设备可加工性、设备维修窗口、跨厂转运合法性和转运时间、
          setup 时间、非组批设备容量、有限组批容量与同族兼容。
        - 如果某个任务当前没有任何可行动作，应将其延后到下一可行时间点、换候选设备、
          换加工路径或放到第二阶段补齐，不能写出非法记录。
        - 如果补齐阶段发现某个已启动任务无法在 qtime 上界内接续，应回滚该任务在第一阶段
          的局部安排，改为整体晚排；不能保留半截非法链。
        - 代码中必须显式实现并使用以下函数名，外层框架会做轻量静态门禁：
          `generate_candidate_actions`、`is_feasible_action`、`score_action`、
          `commit_action`、`internal_validate_solution`。
        - `commit_action` 是唯一允许改变调度状态和写入工序记录的入口；其他逻辑只能生成
          候选动作、过滤动作或计算评分，不能直接向最终 solution 写工序。
        - `internal_validate_solution` 在写出 JSON 前执行一致性自检：每个任务只选择一条路径，
          每个选中路径工序只出现一次，前驱存在，基础 qtime/维修/转运/组批检查通过；
          自检失败时应回滚或重排，不能把失败解直接写出。

        ### 第一阶段：窗口内产量优先
        - 目标是在 horizon 内完成尽可能高权重的任务，而不是先追求所有任务都入窗。
        - 候选选择应优先考虑任务权重、剩余工艺链长度、关键设备负载、setup 同族连续、
          组批可合批性、释放时间、维修避让和跨厂转运。
        - 一阶段不能只排某个任务的前半段工序后把后续工序丢给很晚的二阶段，否则容易
          违反最大生产间隔。若一个任务已经启动，后续工序必须在 qtime 上界内紧凑接续；
          如果无法闭环，应把该任务整体转入二阶段作为晚完工任务。
        - 对可能在 horizon 前完成的任务，应尽量按完整工艺链评估插入，而不是孤立评估
          单道工序的局部最早开工。
        - 每个任务在选中路径下的每道工序只能输出一次；不能为了补全而重复输出工序，
          也不能输出不属于选中路径的额外工序。

        ### 第二阶段：补全完整合法解
        - 在一阶段结束后，必须补齐所有未排任务和未排工序，最终输出包含全部任务、全部
          选中路径工序的完整解。
        - 二阶段的优化目标不是产量，而是合法性和稳定性：满足前后序、释放时间、
          最小/最大生产间隔、维修窗口、setup、跨厂转运和有限组批容量。
        - 二阶段应优先采用紧凑排程：从任务释放时间或上一道工序完成时间出发，尽量连续
          安排完整后续链，避免因等待过长触发 qtime 上界错误。
        - 如果一阶段留下了已启动但未闭环的任务，二阶段必须优先修复这些任务，再处理
          完全未启动的任务。
        - qtime 不是只约束相邻工序。安排任意一道工序时，必须遍历该任务所选路径下所有
          以该工序为 `end_process_seq` 的 qtime 记录，同时满足：
          1. `lower time bound`：当前工序的指定时间点不能早于起点工序指定时间点
             加最小生产间隔；
          2. `upper time bound`：当前工序的指定时间点不能晚于起点工序指定时间点
             加最大生产间隔。
        - 如果为了满足 qtime 下界而推迟某道工序，必须把同一任务后续工序一起向后传播；
          如果传播后违反 qtime 上界，应重新选择该任务的插入位置/设备/路径，而不是输出非法解。
        - 组批工序在两个阶段都必须遵守有限批容量；同一批的 machine/start/finish/family
          和批时长必须与校验器口径一致。

        ### 最低验收逻辑
        - 即使一阶段产量较低，二阶段也必须保证 `fully_scheduled_tasks == total_tasks`。
        - 不能为了提高窗口内产量牺牲完整性；非法高产候选没有意义。
        - 如果上一轮错误主要是 lower_time_bound、upper_time_bound、missing_predecessor、
          路径缺工序或 scheduled_ops 数量异常，下一轮优先修复二阶段补链和 qtime 传播逻辑，
          而不是继续调 setup/产量权重。
        """
    ).strip()


def build_generation_prompt(
    docs: list[tuple[Path, str]],
    previous_code: str | None,
    previous_result: ValidationResult | None,
    experience_memory: str,
    best_artifact: BestArtifact | None,
    best_attempt: BestAttempt | None,
    analyzer_advice: str,
    prompt_mode: str,
    round_index: int,
    candidate_index: int,
    candidates_per_iter: int,
    max_presets_per_solver: int,
    target_weight: float,
    target_setup: int,
    previous_code_chars: int = 28000,
) -> str:
    """构造发送给 DeepSeek 的用户 Prompt。"""

    doc_blocks = []
    for path, text in docs:
        doc_blocks.append(f"## 文档：{path.name}\n\n{text}")

    feedback = ""
    if previous_result is not None:
        feedback = textwrap.dedent(
            f"""
            ## 上一轮运行与校验反馈

            - iteration: {previous_result.iteration}
            - solver_exit_code: {previous_result.solver_exit_code}
            - validator_exit_code: {previous_result.validator_exit_code}
            - metrics: {json.dumps(previous_result.metrics, ensure_ascii=False)}
            - error_count: {previous_result.error_count}

            ### solver stdout
            ```text
            {previous_result.solver_stdout[-6000:]}
            ```

            ### solver stderr
            ```text
            {previous_result.solver_stderr[-6000:]}
            ```

            ### validator stdout
            ```text
            {previous_result.validator_stdout[-9000:]}
            ```

            ### validator stderr
            ```text
            {previous_result.validator_stderr[-6000:]}
            ```
            """
        ).strip()

    code_block = ""
    if previous_code and previous_code_chars != 0:
        limit = max(1000, previous_code_chars)
        code_block = f"## 上一轮 deepseek_solver.py\n\n```python\n{previous_code[-limit:]}\n```"

    best_block = ""
    if best_artifact is not None:
        best_block = textwrap.dedent(
            f"""
            ## 当前最佳合法结果

            - iteration: {best_artifact.iteration}
            - completed_weight_within_horizon: {best_artifact.completed_weight}
            - setup_count_positive: {best_artifact.setup_count}
            - score: {best_artifact.score}
            - metrics: {json.dumps(best_artifact.metrics, ensure_ascii=False)}
            """
        ).strip()

    best_attempt_block = ""
    if best_artifact is None and best_attempt is not None:
        best_attempt_block = textwrap.dedent(
            f"""
            ## 当前最佳未合法尝试（防退化基线）

            目前还没有完整合法解。以下是不合法候选中最好的一个，下一轮应优先在此基础上修复，
            不要退化为更少完整任务或更多错误：

            - iteration: {best_attempt.iteration}
            - fully_scheduled_tasks: {best_attempt.fully_scheduled_tasks}/{best_attempt.total_tasks}
            - error_count: {best_attempt.error_count}
            - completed_weight_within_horizon: {best_attempt.completed_weight}
            - setup_count_positive: {best_attempt.setup_count}
            - metrics: {json.dumps(best_attempt.metrics, ensure_ascii=False)}
            """
        ).strip()

    memory_block = ""
    if experience_memory.strip():
        memory_block = f"## 历史经验与失败模式\n\n{experience_memory[-12000:]}"

    advice_block = ""
    if analyzer_advice.strip():
        advice_block = textwrap.dedent(
            f"""
            ## Analyzer Agent 的结构化修复建议

            下面建议由另一次 DeepSeek 调用基于校验报告、候选代码和历史指标生成。
            请优先执行这些建议；若建议之间冲突，优先保证完整合法。

            {analyzer_advice[-12000:]}
            """
        ).strip()

    context_summary = (
        build_wrapper_context_summary()
        if prompt_mode == "wrapper"
        else build_strict_context_summary()
    )
    mode_title = (
        "文档+校验器严格模式"
        if prompt_mode == "strict"
        else "现有求解器接口复用模式"
    )

    return textwrap.dedent(
        f"""
        你需要生成一个完整的 Python 文件 `deepseek_solver.py`，用于复杂 FJSP 有限组批调度。

        当前实验模式：{mode_title}。
        请基于下面三份文档、允许信息和上一轮反馈，直接输出完整 Python 源码。
        不要输出 Markdown 解释；如果必须使用代码块，只能包含一个 python 代码块。

        ## 优化目标
        - 当前为第 {round_index + 1} 轮、第 {candidate_index + 1}/{candidates_per_iter} 个候选。
        1. 生成合法完整解，必须包含所有任务。
        2. 校验口径为有限组批，必须通过 `scripts/validate_batch_solution.py`。
        3. 在合法前提下，尽量使 `completed_weight_within_horizon >= {target_weight}`。
        4. 若产量接近或达到目标，尽量使 `setup_count_positive <= {target_setup}`。

        ## 允许信息与边界
        {context_summary}

        {build_internal_two_stage_requirements()}

        ## 代码实现建议
        - 严格按当前实验模式的边界实现；若是严格模式，必须自行实现启发式调度逻辑。
        - 可以内置少量启发式策略或参数，例如候选工序排序、路径选择、组批等待、同族优先、setup 惩罚、尾部补齐等，但这些规则和参数需要你根据文档、校验反馈和经验自行提出。
        - 必须把硬约束写成动作过滤逻辑，而不是写成评分惩罚。推荐代码结构为：
          `ready_ops -> candidate actions -> is_feasible_action -> score_action -> commit_action -> update_state`。
        - `is_feasible_action` 返回 False 的动作不能被调度；不能为了保证输出完整而用占位时间、
          默认机器或重复工序强行补齐。
        - 为便于外层框架识别，生成代码必须定义并调用：
          `generate_candidate_actions`、`is_feasible_action`、`score_action`、
          `commit_action`、`internal_validate_solution`。
        - 单个 `deepseek_solver.py` 内最多评估 {max_presets_per_solver} 组 preset；外层框架会负责多轮、多候选搜索。
        - 如果代码中出现名为 `presets` 的列表，该列表长度必须 <= {max_presets_per_solver}，否则外层框架会直接拒绝候选。
        - 首要目标是先在限定时间内产出一个可校验解，不要把大量搜索塞进单个 solver。
        - 可以在 solver 内部比较多个候选的 `scheduler.metrics()`，但最终合法性仍由外层框架调用校验器确认。
        - 不要在生成的 solver 中调用 subprocess、网络 API 或删除文件；外层框架会负责校验和迭代。
        - 输出文件必须是命令行 `--output` 指定的路径。
        - 要注意 Windows 路径和 UTF-8 中文文件名。

        ## 本轮自我改进要求
        - 先根据历史经验判断上一轮失败或不足的原因。
        - 若上一轮非法，优先修复代码错误、输出格式、有限组批合法性和完整性。
        - 若上一轮合法但指标不佳，修改启发式结构或候选 preset，不要只做无意义微调。
        - 保留已经有效的策略，避免反复丢失当前最佳合法结果。
        - 同一轮多个候选应尽量保持差异，例如一个候选偏高产量、一个候选偏低 setup、一个候选偏有限组批稳定性。

        {feedback}

        {best_block}

        {best_attempt_block}

        {advice_block}

        {memory_block}

        {code_block}

        {chr(10).join(doc_blocks)}
        """
    ).strip()


def build_analyzer_prompt(
    *,
    docs: list[tuple[Path, str]],
    previous_code: str | None,
    previous_result: ValidationResult | None,
    experience_memory: str,
    best_artifact: BestArtifact | None,
    best_attempt: BestAttempt | None,
    prompt_mode: str,
) -> str:
    """构造二阶段 Analyzer 的反思 Prompt。

    Analyzer 不写代码，只读报告和候选代码，输出下一轮 Coder 可以执行的
    结构化修改建议。
    """

    doc_summary = "\n".join(f"- {path.name}" for path, _ in docs)
    result_block = ""
    if previous_result is not None:
        report_text = ""
        if previous_result.report_path.exists():
            report_text = previous_result.report_path.read_text(
                encoding="utf-8", errors="replace"
            )
        result_block = textwrap.dedent(
            f"""
            ## 上一轮运行结果

            - solver_exit_code: {previous_result.solver_exit_code}
            - validator_exit_code: {previous_result.validator_exit_code}
            - error_count: {previous_result.error_count}
            - metrics: {json.dumps(previous_result.metrics, ensure_ascii=False)}

            ### validator report JSON / sampled errors
            ```json
            {report_text[-18000:]}
            ```

            ### solver stdout tail
            ```text
            {previous_result.solver_stdout[-5000:]}
            ```

            ### solver stderr tail
            ```text
            {previous_result.solver_stderr[-5000:]}
            ```
            """
        ).strip()

    best_block = ""
    if best_artifact is not None:
        best_block = json.dumps(
            {
                "type": "best_legal",
                "iteration": best_artifact.iteration,
                "metrics": best_artifact.metrics,
                "score": best_artifact.score,
            },
            ensure_ascii=False,
            indent=2,
        )
    elif best_attempt is not None:
        best_block = json.dumps(
            {
                "type": "best_invalid_attempt",
                "iteration": best_attempt.iteration,
                "fully_scheduled_tasks": best_attempt.fully_scheduled_tasks,
                "total_tasks": best_attempt.total_tasks,
                "error_count": best_attempt.error_count,
                "metrics": best_attempt.metrics,
                "report": str(best_attempt.report_path),
            },
            ensure_ascii=False,
            indent=2,
        )

    code_block = ""
    if previous_code:
        code_block = f"## 上一轮候选代码\n\n```python\n{previous_code[-24000:]}\n```"

    return textwrap.dedent(
        f"""
        你是复杂 FJSP 调度代码的 Analyzer Agent。你不写完整代码，只负责诊断上一轮
        `deepseek_solver.py` 为什么没有通过有限组批校验，并提出下一轮 Coder Agent
        可以执行的修复方案。

        当前实验模式：{prompt_mode}
        可用文档：{doc_summary}

        重要边界：
        - 如果是 strict 模式，下一轮 Coder 仍不能调用项目已有求解器，只能自行实现调度逻辑。
        - 校验器是唯一裁判，不能建议放宽约束或修改校验器。
        - 优先目标是得到完整合法解，其次才是产量和 setup。
        - 诊断时请按“内部两阶段调度”视角检查：第一阶段是否以可闭环任务链抢产，
          第二阶段是否优先补齐已启动未闭环任务并紧凑补全所有剩余工序。
        - 重点检查候选代码是否真的实现了硬约束动作过滤。如果代码是先生成排程、再在末尾
          修补或依赖外部校验器发现错误，应明确指出这是根因，并要求下一轮改成
          `generate feasible actions -> filter hard constraints -> score -> commit action`。
        - `lower time bound` 通常表示 qtime 最小间隔未满足，不要直接假设是释放时间
          或绝对/相对时间基准错误；必须结合 qtime、前驱记录和当前工序时间判断。
        - `scheduled_ops` 明显高于选中路径总工序数时，优先怀疑补全过程重复输出工序
          或输出了非选中路径工序。

        请输出结构化 Markdown，必须包含：
        1. `主要错误类型`：按影响程度排序，给出错误数量或证据。
        2. `根因判断`：指出候选代码中的具体逻辑缺陷。
        3. `下一轮修改指令`：用可执行的工程语言告诉 Coder 如何改。
        4. `禁止事项`：哪些改法会导致退化，必须避免。
        5. `验收标准`：下一轮至少要达到什么指标，例如 error_count 下降、完整任务不下降。

        ## 当前最佳基线
        ```json
        {best_block}
        ```

        ## 历史经验
        ```text
        {experience_memory[-12000:]}
        ```

        {result_block}

        {code_block}
        """
    ).strip()


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
    wire_api: str = "chat",
    timeout: int = 180,
) -> str:
    """调用大模型接口并返回纯文本响应。

    默认沿用 DeepSeek 的 Chat Completions 协议。部分 Codex 代理服务虽然
    也暴露 OpenAI 兼容地址，但实际要求使用 Responses API；因此这里通过
    `wire_api` 做最小分流，避免改动上层自迭代框架。
    """

    resolved_system_prompt = system_prompt or (
        "你是资深组合优化与生产调度工程师。你只输出可运行 Python 代码，"
        "并严格遵守用户提供的校验器口径。"
    )

    if wire_api == "responses":
        endpoint = base_url.rstrip("/") + "/responses"
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": resolved_system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "store": False,
        }
    else:
        endpoint = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": resolved_system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

    raw = _post_json(
        endpoint=endpoint,
        payload=payload,
        api_key=api_key,
        timeout=timeout,
    )

    data = json.loads(raw)
    if wire_api == "responses":
        texts: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return "\n".join(texts)
        raise RuntimeError(f"Unexpected Responses API response: {raw[:1200]}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected DeepSeek response: {raw[:1200]}") from exc


def _post_json(*, endpoint: str, payload: dict[str, Any], api_key: str, timeout: int) -> str:
    """发送 JSON POST 请求。

    常规情况下使用 Python 标准库，便于跨平台运行。若 Windows 上的 Python
    TLS 栈与某些代理服务握手失败，则回退到系统 curl，并通过临时 config
    文件传入鉴权头，避免把 API Key 暴露到命令行参数或项目文件。
    """

    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        if sys.platform.startswith("win") and shutil.which("curl.exe"):
            return _post_json_with_curl(
                endpoint=endpoint,
                payload=request_body,
                api_key=api_key,
                timeout=timeout,
            )
        raise RuntimeError(f"DeepSeek request failed: {exc}") from exc


def _curl_quote(value: str) -> str:
    """转义 curl config 文件中的双引号和反斜杠。"""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def _post_json_with_curl(
    *, endpoint: str, payload: bytes, api_key: str, timeout: int
) -> str:
    """Windows 兜底请求实现：用 curl 处理证书吊销检查和 TLS 兼容性。"""

    body_path: Path | None = None
    config_path: Path | None = None
    try:
        body_fd, body_name = tempfile.mkstemp(prefix="fjsp_llm_body_", suffix=".json")
        config_fd, config_name = tempfile.mkstemp(prefix="fjsp_llm_curl_", suffix=".cfg")
        os.close(body_fd)
        os.close(config_fd)
        body_path = Path(body_name)
        config_path = Path(config_name)
        body_path.write_bytes(payload)
        config_text = "\n".join(
            [
                f'url = "{_curl_quote(endpoint)}"',
                'request = "POST"',
                f'header = "Authorization: Bearer {_curl_quote(api_key)}"',
                'header = "Content-Type: application/json"',
                f'data-binary = "@{_curl_quote(body_path.as_posix())}"',
                f"max-time = {int(timeout)}",
                "silent",
                "show-error",
                "ssl-no-revoke",
                'write-out = "\\n__HTTP_STATUS__:%{http_code}\\n"',
            ]
        )
        config_path.write_text(config_text, encoding="utf-8")
        completed = subprocess.run(
            ["curl.exe", "--config", str(config_path)],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout + 10,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"curl fallback failed with code {completed.returncode}: "
                f"{completed.stderr[:1200]}"
            )
        raw = completed.stdout
        marker = "\n__HTTP_STATUS__:"
        if marker not in raw:
            return raw
        body, status_text = raw.rsplit(marker, 1)
        status = int(status_text.strip().splitlines()[0])
        if status >= 400:
            raise RuntimeError(f"DeepSeek HTTP {status}: {body[:1200]}")
        return body
    finally:
        for path in (body_path, config_path):
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass


def extract_python_code(response_text: str) -> str:
    """从模型响应中提取 Python 代码。"""

    matches = re.findall(r"```(?:python|py)?\s*(.*?)```", response_text, flags=re.S | re.I)
    if matches:
        code = max(matches, key=len)
    else:
        code = response_text
    return code.strip() + "\n"


def count_literal_presets(code: str) -> int | None:
    """粗略统计生成代码中 `presets = [...]` 的字面量候选数。"""

    match = re.search(r"presets\s*=\s*\[(.*?)\n\s*\]", code, flags=re.S)
    if not match:
        return None
    body = match.group(1)
    names = re.findall(r"['\"]name['\"]\s*:", body)
    if names:
        return len(names)
    # 兜底：统计顶层字典开头。这个粗略检查只用于性能护栏。
    return len(re.findall(r"\{\s*['\"]", body))


def check_generated_code_safety(
    code: str,
    max_presets_per_solver: int,
    prompt_mode: str,
) -> list[str]:
    """对生成代码做轻量静态安全检查。"""

    lowered = code.lower()
    issues = []
    for pattern in FORBIDDEN_CODE_PATTERNS:
        if pattern.lower() in lowered:
            issues.append(f"forbidden pattern found: {pattern}")
    if prompt_mode == "strict":
        strict_forbidden = (
            "rl_relaxed_solver",
            "RelaxedRLScheduler",
            "build_instance",
            "SetupRowStore",
            "dump_solution",
            "solve_best_preset",
            "batch_parameter_tune",
            "auto_tune_batch_parameters",
            "validate_batch_solution",
            "scripts.",
            "from scripts",
            "outputs/",
            "outputs\\",
        )
        for pattern in strict_forbidden:
            if pattern.lower() in lowered:
                issues.append(f"strict mode forbids project reuse/reference: {pattern}")
        required_action_filter_functions = (
            "generate_candidate_actions",
            "is_feasible_action",
            "score_action",
            "commit_action",
            "internal_validate_solution",
        )
        for name in required_action_filter_functions:
            if not re.search(rf"def\s+{name}\s*\(", code):
                issues.append(f"strict mode requires action-filter function: {name}")
            if len(re.findall(rf"\b{name}\s*\(", code)) < 2:
                issues.append(f"strict mode requires {name} to be called, not only defined")
        direct_solution_writes = len(
            re.findall(r"solution\s*\[\s*['\"]task['\"]\s*\]", code)
        )
        if direct_solution_writes > 6:
            issues.append(
                "too many direct writes to solution['task']; route schedule writes "
                "through one commit_action path and avoid repair-time bypasses"
            )
    if "if __name__" not in code:
        issues.append("missing if __name__ entrypoint")
    if "argparse" not in code:
        issues.append("missing argparse CLI")
    preset_count = count_literal_presets(code)
    if preset_count is not None and preset_count > max_presets_per_solver:
        issues.append(
            f"too many literal presets: {preset_count} > {max_presets_per_solver}; "
            "generate fewer candidates and let the outer framework iterate"
        )
    return issues


def metric_float(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    """安全读取浮点指标。"""

    try:
        return float(metrics.get(name, default))
    except (TypeError, ValueError):
        return default


def metric_int(metrics: dict[str, Any], name: str, default: int = 0) -> int:
    """安全读取整数指标。"""

    try:
        return int(metrics.get(name, default))
    except (TypeError, ValueError):
        return default


def is_complete_legal(result: ValidationResult) -> bool:
    """判断一次候选是否为完整合法解。"""

    if result.error_count != 0 or result.validator_exit_code != 0:
        return False
    metrics = result.metrics
    total_tasks = metric_int(metrics, "total_tasks", -1)
    fully_scheduled = metric_int(metrics, "fully_scheduled_tasks", -2)
    if total_tasks >= 0 and fully_scheduled != total_tasks:
        return False
    return True


def score_result(result: ValidationResult, setup_penalty: float) -> float:
    """把合法候选映射为单标量分数，便于自动保留 best。"""

    if not is_complete_legal(result):
        return float("-inf")
    weight = metric_float(result.metrics, "completed_weight_within_horizon")
    setup = metric_int(result.metrics, "setup_count_positive")
    return weight - setup_penalty * setup


def attempt_key(result: ValidationResult) -> tuple[float, int, float]:
    """非法候选之间的比较键。

    先看完整排程比例，再看错误数少，再看窗口内产量高。这样可以避免
    LLM 为了提高产量而重新漏排大量任务。
    """

    metrics = result.metrics
    total = max(1, metric_int(metrics, "total_tasks", 1))
    fully = metric_int(metrics, "fully_scheduled_tasks", 0)
    error_count = result.error_count
    weight = metric_float(metrics, "completed_weight_within_horizon")
    return (fully / total, -error_count, weight)


def build_experience_entry(
    *,
    result: ValidationResult,
    score: float,
    best_artifact: BestArtifact | None,
) -> str:
    """根据运行/校验证据生成可喂给下一轮的经验记录。"""

    metrics = result.metrics
    weight = metric_float(metrics, "completed_weight_within_horizon")
    setup = metric_int(metrics, "setup_count_positive", -1)
    fully = metric_int(metrics, "fully_scheduled_tasks", -1)
    total = metric_int(metrics, "total_tasks", -1)

    log_text = "\n".join(
        [
            result.solver_stdout,
            result.solver_stderr,
            result.validator_stdout,
            result.validator_stderr,
        ]
    )
    diagnostics: list[str] = []

    if "output misses" in log_text:
        diagnostics.append(
            "完整性错误：输出必须包含全部任务；每个任务必须选择一条路径，并输出该路径下所有工序。"
        )
    if "process seqs do not cover selected path" in log_text:
        diagnostics.append(
            "路径工序错误：同一任务输出的 seq 列表必须与选中 path 的 process_list 完全一致，不能只输出尾部或跳过中间工序。"
        )
    if "violates lower time bound" in log_text:
        diagnostics.append(
            "最小间隔错误：qtime_info 不是只约束相邻工序。调度每道工序时，应遍历该 path 下所有 qtime 记录；若当前工序是 end_process_seq，必须根据 start_process_type 取起点工序 start/end，根据 end_process_type 约束当前工序 start/end，使 end_point >= start_point + min_process_interval。"
        )
    if "violates upper time bound" in log_text:
        diagnostics.append(
            "最大间隔错误：同样需要遍历所有 qtime 记录，并保证 end_point <= start_point + max_process_interval；若无法满足，应推迟/重排前序链，而不是忽略约束。"
        )
    if "overlaps maintenance" in log_text:
        diagnostics.append(
            "维修窗口错误：任何工序区间都必须与设备维修窗口不相交；若调整开始时间避让维修，需重新计算 setup、转运和后续 qtime。"
        )
    if "missing valid predecessor record" in log_text:
        diagnostics.append(
            "前序记录错误：后续工序只有在前序工序成功排程并记录后才能排程；维修或候选设备调整失败时不能留下半条任务链。"
        )
    if "illegal factory transfer" in log_text:
        diagnostics.append(
            "跨厂转运错误：相邻工序若设备所属厂区不同，必须确认输入 transition 矩阵允许该设备对设备转运，并把转运时间加入后序工序最早开工；没有合法转运关系时应换候选设备或换路径。"
        )
    if "batch duration" in log_text or "batch family" in log_text or "batch size" in log_text:
        diagnostics.append(
            "有限组批错误：同一批必须 machine/start/finish 相同、family 兼容、批大小不超过 curr_batch_size，批时长等于批内最大单件加工时长。"
        )

    if is_complete_legal(result):
        if best_artifact is None or score > best_artifact.score:
            verdict = "合法完整，且刷新当前综合分。下一轮应保留本轮核心策略，并在其附近扩展候选。"
        else:
            verdict = "合法完整，但未刷新综合分。下一轮应分析其相对 best 的产量/setup 取舍，避免退化。"
    elif result.solver_exit_code not in (0, None):
        verdict = "solver 运行失败。下一轮优先修复 Python 异常、导入路径、CLI 参数或输出文件生成。"
    elif result.validator_exit_code not in (0, None):
        verdict = "solver 能输出解，但有限组批校验失败。下一轮优先修复校验器报告中的约束错误。"
    elif not result.solution_path.exists():
        verdict = "未生成解文件。下一轮必须保证按 --output 写出标准 JSON。"
    else:
        verdict = "运行或校验未完成。下一轮降低复杂度，先生成可运行合法 baseline。"

    validator_tail = result.validator_stdout[-3000:].strip()
    solver_tail = (result.solver_stderr or result.solver_stdout)[-2000:].strip()

    return textwrap.dedent(
        f"""
        ### Iteration {result.iteration}

        - solver_exit_code: {result.solver_exit_code}
        - validator_exit_code: {result.validator_exit_code}
        - error_count: {result.error_count}
        - completed_weight_within_horizon: {weight}
        - setup_count_positive: {setup}
        - fully_scheduled_tasks: {fully}
        - total_tasks: {total}
        - scalar_score: {score}
        - 经验判断: {verdict}

        结构化诊断：
        {chr(10).join("- " + item for item in diagnostics) if diagnostics else "- 暂无可归纳诊断；优先读取校验器错误明细。"}

        关键日志摘录：
        ```text
        {solver_tail}
        {validator_tail}
        ```
        """
    ).strip()


def append_text(path: Path, text: str) -> None:
    """追加写入 UTF-8 文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n\n")


def _path_from_json(value: Any) -> Path:
    """把 JSON 中记录的路径恢复为 Path；空值返回空 Path。"""

    if value is None:
        return Path("")
    return Path(str(value))


def _same_path_text(left: Path, right: Path) -> bool:
    """按文本归一化比较路径，避免历史报告中绝对/相对路径格式差异。"""

    if not str(left) or not str(right):
        return False
    return str(left).replace("\\", "/").lower() == str(right).replace("\\", "/").lower()


def _best_attempt_from_summary(summary_path: Path) -> BestAttempt | None:
    """从 best_attempt_summary.json 恢复最佳非法/部分合法样本。"""

    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None

    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    error_count = int(summary.get("error_count", metrics.get("error_count", 999999)))
    synthetic_result = ValidationResult(
        iteration=int(summary.get("iteration", -1)),
        solver_exit_code=None,
        validator_exit_code=0 if error_count == 0 else 1,
        solution_path=_path_from_json(summary.get("solution")),
        report_path=_path_from_json(summary.get("report")),
        metrics=metrics,
        error_count=error_count,
        solver_stdout="",
        solver_stderr="",
        validator_stdout="",
        validator_stderr="",
    )
    return BestAttempt(
        iteration=int(summary.get("iteration", -1)),
        attempt_key=attempt_key(synthetic_result),
        fully_scheduled_tasks=int(
            summary.get(
                "fully_scheduled_tasks", metric_int(metrics, "fully_scheduled_tasks")
            )
        ),
        total_tasks=int(summary.get("total_tasks", metric_int(metrics, "total_tasks"))),
        error_count=error_count,
        completed_weight=float(
            summary.get(
                "completed_weight_within_horizon",
                metric_float(metrics, "completed_weight_within_horizon"),
            )
        ),
        setup_count=int(
            summary.get("setup_count_positive", metric_int(metrics, "setup_count_positive"))
        ),
        solver_candidate_path=_path_from_json(summary.get("solver")),
        solution_path=synthetic_result.solution_path,
        report_path=synthetic_result.report_path,
        metrics=metrics,
    )


def validation_result_from_best_attempt(best_attempt: BestAttempt) -> ValidationResult:
    """把最佳未合法样本转换成 Analyzer 可复盘的上一轮结果。"""

    return ValidationResult(
        iteration=best_attempt.iteration,
        solver_exit_code=None,
        validator_exit_code=0 if best_attempt.error_count == 0 else 1,
        solution_path=best_attempt.solution_path,
        report_path=best_attempt.report_path,
        metrics=best_attempt.metrics,
        error_count=best_attempt.error_count,
        solver_stdout="",
        solver_stderr="",
        validator_stdout="",
        validator_stderr="",
    )


def best_attempt_from_reports(output_dir: Path) -> BestAttempt | None:
    """从历史校验报告恢复最佳未合法尝试。"""

    best: BestAttempt | None = _best_attempt_from_summary(
        output_dir / "best_attempt_summary.json"
    )
    reports_dir = output_dir / "reports"
    if not reports_dir.exists():
        return best
    for report_path in sorted(reports_dir.glob("deepseek_validate_iter*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        metrics = report.get("metrics", {})
        error_count = int(report.get("error_count", metrics.get("error_count", 999999)))
        result = ValidationResult(
            iteration=-1,
            solver_exit_code=None,
            validator_exit_code=0 if error_count == 0 else 1,
            solution_path=Path(str(report.get("solution", ""))),
            report_path=report_path,
            metrics=metrics,
            error_count=error_count,
            solver_stdout="",
            solver_stderr="",
            validator_stdout="",
            validator_stderr="",
        )
        key = attempt_key(result)
        match = re.search(r"iter(\d+)", report_path.stem)
        iteration = int(match.group(1)) if match else -1
        candidate = BestAttempt(
            iteration=iteration,
            attempt_key=key,
            fully_scheduled_tasks=metric_int(metrics, "fully_scheduled_tasks"),
            total_tasks=metric_int(metrics, "total_tasks"),
            error_count=error_count,
            completed_weight=metric_float(metrics, "completed_weight_within_horizon"),
            setup_count=metric_int(metrics, "setup_count_positive"),
            solver_candidate_path=(
                best.solver_candidate_path
                if best is not None and _same_path_text(best.report_path, report_path)
                else Path("")
            ),
            solution_path=Path(str(report.get("solution", ""))),
            report_path=report_path,
            metrics=metrics,
        )
        if best is None or candidate.attempt_key > best.attempt_key:
            best = candidate
    return best


def next_run_index(output_dir: Path) -> int:
    """根据已保存报告推断下一次候选编号，避免续跑覆盖历史记录。"""

    reports_dir = output_dir / "reports"
    if not reports_dir.exists():
        return 0
    max_index = -1
    for pattern in ("framework_summary_iter*.json", "framework_reject_iter*.json"):
        for path in reports_dir.glob(pattern):
            match = re.search(r"iter(\d+)", path.stem)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def run_command(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    """运行外部命令并捕获文本输出。"""

    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def run_solver_and_validator(
    *,
    root: Path,
    solver_path: Path,
    input_path: Path,
    output_dir: Path,
    iteration: int,
    solver_timeout: int,
    validator_timeout: int,
) -> ValidationResult:
    """运行生成的 solver，并用有限组批校验器验证候选解。"""

    solution_path = output_dir / "solutions" / f"deepseek_solution_iter{iteration:02d}.json"
    report_path = output_dir / "reports" / f"deepseek_validate_iter{iteration:02d}.json"
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    solver_cmd = [
        sys.executable,
        str(solver_path),
        "--root",
        str(root),
        "--input",
        str(input_path),
        "--output",
        str(solution_path),
    ]
    try:
        solver_proc = run_command(solver_cmd, root, timeout=solver_timeout)
    except subprocess.TimeoutExpired as exc:
        return ValidationResult(
            iteration=iteration,
            solver_exit_code=None,
            validator_exit_code=None,
            solution_path=solution_path,
            report_path=report_path,
            metrics={},
            error_count=999999,
            solver_stdout=exc.stdout or "",
            solver_stderr=(exc.stderr or "") + "\n[framework] solver timeout",
            validator_stdout="",
            validator_stderr="",
        )

    validator_stdout = ""
    validator_stderr = ""
    validator_exit_code: int | None = None
    metrics: dict[str, Any] = {}
    error_count = 999999

    if solution_path.exists():
        validator_cmd = [
            sys.executable,
            str(root / "scripts" / "validate_batch_solution.py"),
            "--input",
            str(input_path),
            "--solution",
            str(solution_path),
            "--report",
            str(report_path),
            "--max-errors",
            "40",
        ]
        try:
            validator_proc = run_command(validator_cmd, root, timeout=validator_timeout)
            validator_exit_code = validator_proc.returncode
            validator_stdout = validator_proc.stdout
            validator_stderr = validator_proc.stderr
        except subprocess.TimeoutExpired as exc:
            validator_stdout = exc.stdout or ""
            validator_stderr = (exc.stderr or "") + "\n[framework] validator timeout"
            validator_exit_code = None

        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
            metrics = report.get("metrics", {})
            error_count = int(report.get("error_count", 999999))
    else:
        validator_stderr = "[framework] solver did not create solution file"

    return ValidationResult(
        iteration=iteration,
        solver_exit_code=solver_proc.returncode,
        validator_exit_code=validator_exit_code,
        solution_path=solution_path,
        report_path=report_path,
        metrics=metrics,
        error_count=error_count,
        solver_stdout=solver_proc.stdout,
        solver_stderr=solver_proc.stderr,
        validator_stdout=validator_stdout,
        validator_stderr=validator_stderr,
    )


def is_target_met(result: ValidationResult, target_weight: float, target_setup: int) -> bool:
    """判断候选结果是否达到本轮实验目标。"""

    metrics = result.metrics
    return (
        is_complete_legal(result)
        and float(metrics.get("completed_weight_within_horizon", 0.0)) >= target_weight
        and int(metrics.get("setup_count_positive", 10**9)) <= target_setup
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """写出 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_final_report(
    *,
    output_dir: Path,
    solver_path: Path,
    prompt_mode: str,
    best_artifact: BestArtifact | None,
    best_attempt: BestAttempt | None,
    max_iters: int,
    target_weight: float,
    target_setup: int,
    target_met: bool,
) -> Path:
    """写出本次自迭代实验的最终报告。"""

    report_path = output_dir / "final_report.md"
    experience_path = output_dir / "experience_log.md"
    experience = ""
    if experience_path.exists():
        experience = experience_path.read_text(encoding="utf-8", errors="replace")

    if best_artifact is None:
        if best_attempt is None:
            best_text = "本次运行未发现完整合法候选，也没有可比较的有效尝试。需要查看各轮 safety/report 日志定位原因。"
        else:
            best_text = textwrap.dedent(
                f"""
                本次运行未发现完整合法候选。最佳未合法尝试如下：

                - best_attempt_iteration: {best_attempt.iteration}
                - fully_scheduled_tasks: {best_attempt.fully_scheduled_tasks}/{best_attempt.total_tasks}
                - error_count: {best_attempt.error_count}
                - completed_weight_within_horizon: {best_attempt.completed_weight}
                - setup_count_positive: {best_attempt.setup_count}
                - attempt_solution: {best_attempt.solution_path}
                - attempt_report: {best_attempt.report_path}
                """
            ).strip()
    else:
        best_text = textwrap.dedent(
            f"""
            - best_iteration: {best_artifact.iteration}
            - completed_weight_within_horizon: {best_artifact.completed_weight}
            - setup_count_positive: {best_artifact.setup_count}
            - scalar_score: {best_artifact.score}
            - best_solver: {output_dir / "best_deepseek_solver.py"}
            - best_solution: {output_dir / "best_solution.json"}
            - best_report: {best_artifact.report_path}
            """
        ).strip()

    report = textwrap.dedent(
        f"""
        # DeepSeek Solver 自迭代实验报告

        ## 运行设置

        - max_iters: {max_iters}
        - prompt_mode: {prompt_mode}
        - target_weight: {target_weight}
        - target_setup: {target_setup}
        - target_met: {target_met}
        - active_solver_path: {solver_path}

        ## 最佳合法结果

        {best_text}

        ## 迭代经验记录

        {experience if experience.strip() else "暂无经验记录。"}
        """
    ).strip() + "\n"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> int:
    """框架主流程。"""

    args = parse_args()
    root = args.root.resolve()
    input_path = resolve_path(root, args.input)
    solver_path = resolve_path(root, args.solver_path)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(root, args.doc)
    previous_code: str | None = None
    previous_result: ValidationResult | None = None
    analyzer_advice = ""
    best_artifact: BestArtifact | None = None
    best_attempt: BestAttempt | None = best_attempt_from_reports(output_dir)
    if best_attempt is not None:
        print(
            "[framework] restored best historical attempt: "
            f"iteration={best_attempt.iteration}, "
            f"fully={best_attempt.fully_scheduled_tasks}/{best_attempt.total_tasks}, "
            f"errors={best_attempt.error_count}, "
            f"weight={best_attempt.completed_weight}",
            flush=True,
        )
    target_met = False
    experience_path = output_dir / "experience_log.md"
    if experience_path.exists():
        experience_memory = experience_path.read_text(encoding="utf-8", errors="replace")
    else:
        experience_memory = ""

    if args.two_stage_reflection and best_attempt is not None:
        # 二阶段模式下，历史最佳非法样本本身就是最有价值的反思材料。
        # 将它恢复成“上一轮结果”，让 Analyzer 首轮就围绕剩余错误诊断，
        # 避免重新从随机低质量候选开始试错。
        previous_result = validation_result_from_best_attempt(best_attempt)
        if best_attempt.solver_candidate_path.exists():
            previous_code = best_attempt.solver_candidate_path.read_text(
                encoding="utf-8", errors="replace"
            )
        print(
            "[framework] seeded analyzer with best historical attempt: "
            f"errors={best_attempt.error_count}, "
            f"weight={best_attempt.completed_weight}, "
            f"solver={best_attempt.solver_candidate_path}",
            flush=True,
        )

    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        print(
            f"[error] environment variable {args.api_key_env} is not set. "
            "For safety, put the DeepSeek key in the environment instead of command args.",
            file=sys.stderr,
        )
        return 2

    run_index = next_run_index(output_dir)
    round_offset = run_index
    for iteration in range(args.max_iters):
        actual_round = round_offset + iteration
        for candidate_index in range(max(1, args.candidates_per_iter)):
            if args.two_stage_reflection and previous_result is not None:
                analyzer_prompt = build_analyzer_prompt(
                    docs=docs,
                    previous_code=previous_code,
                    previous_result=previous_result,
                    experience_memory=experience_memory,
                    best_artifact=best_artifact,
                    best_attempt=best_attempt,
                    prompt_mode=args.prompt_mode,
                )
                analyzer_path = output_dir / "analysis" / (
                    f"round_{actual_round:02d}_cand_{candidate_index:02d}_analyzer_prompt.md"
                )
                analyzer_path.parent.mkdir(parents=True, exist_ok=True)
                analyzer_path.write_text(analyzer_prompt, encoding="utf-8")
                print(f"[framework] analyzer prompt saved: {analyzer_path}", flush=True)
                if args.skip_analyzer or args.dry_run:
                    print(
                        "[framework] analyzer skipped; coder will use previous code/result directly",
                        flush=True,
                    )
                else:
                    analyzer_advice = call_deepseek(
                        api_key=api_key,
                        base_url=args.base_url,
                        model=args.model,
                        prompt=analyzer_prompt,
                        temperature=min(args.temperature, 0.2),
                        max_tokens=min(args.max_tokens, 4096),
                        system_prompt=(
                            "你是严谨的组合优化调试分析员。你只输出诊断和修复建议，"
                            "不输出完整代码。"
                        ),
                        wire_api=args.wire_api,
                        timeout=args.llm_timeout,
                    )
                    advice_path = output_dir / "analysis" / (
                        f"round_{actual_round:02d}_cand_{candidate_index:02d}_advice.md"
                    )
                    advice_path.write_text(analyzer_advice, encoding="utf-8")
                    print(f"[framework] analyzer advice saved: {advice_path}", flush=True)

            prompt = build_generation_prompt(
                docs=docs,
                previous_code=previous_code,
                previous_result=previous_result,
                experience_memory=experience_memory,
                best_artifact=best_artifact,
                best_attempt=best_attempt,
                analyzer_advice=analyzer_advice,
                prompt_mode=args.prompt_mode,
                round_index=actual_round,
                candidate_index=candidate_index,
                candidates_per_iter=max(1, args.candidates_per_iter),
                max_presets_per_solver=max(1, args.max_presets_per_solver),
                target_weight=args.target_weight,
                target_setup=args.target_setup,
                previous_code_chars=args.previous_code_chars,
            )
            prefix = f"round_{actual_round:02d}_cand_{candidate_index:02d}"
            prompt_path = output_dir / "prompts" / f"{prefix}_prompt.md"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
            print(f"[framework] prompt saved: {prompt_path}", flush=True)

            if args.dry_run:
                print("[framework] dry-run enabled; stop before DeepSeek API call", flush=True)
                return 0

            response_text = call_deepseek(
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
            response_path.write_text(response_text, encoding="utf-8")
            print(f"[framework] response saved: {response_path}", flush=True)

            code = extract_python_code(response_text)
            candidate_path = output_dir / "candidates" / f"deepseek_solver_{prefix}.py"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(code, encoding="utf-8")
            print(f"[framework] candidate solver saved: {candidate_path}", flush=True)

            issues = check_generated_code_safety(
                code,
                max_presets_per_solver=max(1, args.max_presets_per_solver),
                prompt_mode=args.prompt_mode,
            )
            if issues and not args.allow_unsafe_code:
                issue_text = "\n".join(f"- {item}" for item in issues)
                previous_result = ValidationResult(
                    iteration=run_index,
                    solver_exit_code=None,
                    validator_exit_code=None,
                    solution_path=output_dir / "solutions" / f"deepseek_solution_iter{run_index:02d}.json",
                    report_path=output_dir / "reports" / f"deepseek_validate_iter{run_index:02d}.json",
                    metrics={},
                    error_count=999999,
                    solver_stdout="",
                    solver_stderr=f"[framework] generated code rejected by safety check:\n{issue_text}",
                    validator_stdout="",
                    validator_stderr="",
                )
                previous_code = code
                rejected_score = float("-inf")
                entry = build_experience_entry(
                    result=previous_result,
                    score=rejected_score,
                    best_artifact=best_artifact,
                )
                append_text(experience_path, entry)
                experience_memory += "\n\n" + entry
                save_json(
                    output_dir / "reports" / f"framework_reject_iter{run_index:02d}.json",
                    {
                        "round": iteration,
                        "actual_round": actual_round,
                        "candidate_index": candidate_index,
                        "iteration": run_index,
                        "status": "rejected_by_safety_check",
                        "issues": issues,
                        "candidate": str(candidate_path),
                    },
                )
                print("[framework] candidate rejected by safety check; ask model to repair next", flush=True)
                run_index += 1
                continue

            solver_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate_path, solver_path)
            print(f"[framework] promoted candidate to: {solver_path}", flush=True)

            if args.skip_run:
                print("[framework] skip-run enabled; solver generated but not executed", flush=True)
                return 0

            result = run_solver_and_validator(
                root=root,
                solver_path=solver_path,
                input_path=input_path,
                output_dir=output_dir,
                iteration=run_index,
                solver_timeout=args.solver_timeout,
                validator_timeout=args.validator_timeout,
            )
            previous_result = result
            previous_code = code
            current_score = score_result(result, args.setup_penalty)

            summary = {
                "round": actual_round,
                "candidate_index": candidate_index,
                "iteration": run_index,
                "solver_exit_code": result.solver_exit_code,
                "validator_exit_code": result.validator_exit_code,
                "solution": str(result.solution_path),
                "report": str(result.report_path),
                "metrics": result.metrics,
                "error_count": result.error_count,
                "complete_legal": is_complete_legal(result),
                "scalar_score": current_score,
            }
            save_json(output_dir / "reports" / f"framework_summary_iter{run_index:02d}.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

            entry = build_experience_entry(
                result=result,
                score=current_score,
                best_artifact=best_artifact,
            )
            append_text(experience_path, entry)
            experience_memory += "\n\n" + entry

            if is_complete_legal(result) and (
                best_artifact is None or current_score > best_artifact.score
            ):
                best_artifact = BestArtifact(
                    iteration=run_index,
                    score=current_score,
                    completed_weight=metric_float(
                        result.metrics, "completed_weight_within_horizon"
                    ),
                    setup_count=metric_int(result.metrics, "setup_count_positive"),
                    solver_candidate_path=candidate_path,
                    solution_path=result.solution_path,
                    report_path=result.report_path,
                    metrics=result.metrics,
                )
                best_solver_path = output_dir / "best_deepseek_solver.py"
                best_solution_path = output_dir / "best_solution.json"
                shutil.copyfile(candidate_path, best_solver_path)
                shutil.copyfile(result.solution_path, best_solution_path)
                save_json(
                    output_dir / "best_summary.json",
                    {
                        "iteration": best_artifact.iteration,
                        "score": best_artifact.score,
                        "completed_weight_within_horizon": best_artifact.completed_weight,
                        "setup_count_positive": best_artifact.setup_count,
                        "solver": str(best_solver_path),
                        "solution": str(best_solution_path),
                        "report": str(best_artifact.report_path),
                        "metrics": best_artifact.metrics,
                    },
                )
                print(f"[framework] new best legal candidate: {best_solver_path}", flush=True)

            current_attempt_key = attempt_key(result)
            if best_attempt is None or current_attempt_key > best_attempt.attempt_key:
                best_attempt = BestAttempt(
                    iteration=run_index,
                    attempt_key=current_attempt_key,
                    fully_scheduled_tasks=metric_int(
                        result.metrics, "fully_scheduled_tasks"
                    ),
                    total_tasks=metric_int(result.metrics, "total_tasks"),
                    error_count=result.error_count,
                    completed_weight=metric_float(
                        result.metrics, "completed_weight_within_horizon"
                    ),
                    setup_count=metric_int(result.metrics, "setup_count_positive"),
                    solver_candidate_path=candidate_path,
                    solution_path=result.solution_path,
                    report_path=result.report_path,
                    metrics=result.metrics,
                )
                save_json(
                    output_dir / "best_attempt_summary.json",
                    {
                        "iteration": best_attempt.iteration,
                        "fully_scheduled_tasks": best_attempt.fully_scheduled_tasks,
                        "total_tasks": best_attempt.total_tasks,
                        "error_count": best_attempt.error_count,
                        "completed_weight_within_horizon": best_attempt.completed_weight,
                        "setup_count_positive": best_attempt.setup_count,
                        "solver": str(best_attempt.solver_candidate_path),
                        "solution": str(best_attempt.solution_path),
                        "report": str(best_attempt.report_path),
                        "metrics": best_attempt.metrics,
                    },
                )
                print("[framework] new best invalid/partial attempt recorded", flush=True)

            if is_target_met(result, args.target_weight, args.target_setup):
                target_met = True
                print("[framework] target met; stop early", flush=True)
                if best_artifact is not None:
                    shutil.copyfile(best_artifact.solver_candidate_path, solver_path)
                final_report = write_final_report(
                output_dir=output_dir,
                solver_path=solver_path,
                prompt_mode=args.prompt_mode,
                best_artifact=best_artifact,
                best_attempt=best_attempt,
                    max_iters=args.max_iters,
                    target_weight=args.target_weight,
                    target_setup=args.target_setup,
                    target_met=target_met,
                )
                print(f"[framework] final report: {final_report}", flush=True)
                return 0

            run_index += 1
            time.sleep(1)

    if best_artifact is not None:
        # 最大轮次结束时，根目录 deepseek_solver.py 应回到最佳合法版本，
        # 而不是停留在最后一轮可能退化或非法的候选。
        shutil.copyfile(best_artifact.solver_candidate_path, solver_path)

    final_report = write_final_report(
        output_dir=output_dir,
        solver_path=solver_path,
        prompt_mode=args.prompt_mode,
        best_artifact=best_artifact,
        best_attempt=best_attempt,
        max_iters=args.max_iters,
        target_weight=args.target_weight,
        target_setup=args.target_setup,
        target_met=target_met,
    )
    print(f"[framework] final report: {final_report}", flush=True)
    print("[framework] max iterations reached; inspect reports for current best evidence", flush=True)
    return 0 if best_artifact is not None else 1


if __name__ == "__main__":
    sys.exit(main())

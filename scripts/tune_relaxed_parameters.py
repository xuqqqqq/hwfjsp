#!/usr/bin/env python3
"""组批无限产能批量调参入口。

等价于调用 batch_parameter_tune.py --track relaxed。
需要全自动搜索时，优先使用 auto_tune_relaxed_parameters.py。
"""
from __future__ import annotations

from batch_parameter_tune import main


if __name__ == "__main__":
    # 固定 relaxed 轨道，避免命令行里重复写 --track relaxed。
    raise SystemExit(main(default_track="relaxed", forced_track="relaxed"))

#!/usr/bin/env python3
"""有限组批自动调参入口。

等价于调用 batch_parameter_tune.py --track finite --auto。
"""
from __future__ import annotations

import sys

from batch_parameter_tune import main


if __name__ == "__main__":
    # 提供固定 finite 轨道的简化入口，避免重复输入 --track finite --auto。
    argv = ["--auto", *sys.argv[1:]]
    raise SystemExit(main(argv, default_track="finite", forced_track="finite"))

#!/usr/bin/env python3
"""组批无限产能自动调参入口。

等价于调用 batch_parameter_tune.py --track relaxed --auto。
"""
from __future__ import annotations

import sys

from batch_parameter_tune import main


if __name__ == "__main__":
    # 给用户保留短命令：无需手写 --track relaxed --auto。
    argv = ["--auto", *sys.argv[1:]]
    raise SystemExit(main(argv, default_track="relaxed", forced_track="relaxed"))

#!/usr/bin/env python3
"""有限组批批量调参入口。

等价于调用 batch_parameter_tune.py --track finite。
需要全自动搜索时，优先使用 auto_tune_batch_parameters.py。
"""
from __future__ import annotations

from batch_parameter_tune import main


if __name__ == "__main__":
    # 固定 finite 轨道，避免用户误把有限组批解用 relaxed 校验口径保存。
    raise SystemExit(main(default_track="finite", forced_track="finite"))

# 2026-04-24 Tuning Notes

Input: `data/data1/实际规模输入数据.json`

Horizon: `24480`

Best previous legal high-output point:

| Solution | Weight | Setup |
| --- | ---: | ---: |
| `pareto_18875_818.json` | 18875.15 | 818 |

New legal high-output point:

| Solution | Weight | Setup | Key parameters |
| --- | ---: | ---: | --- |
| `pareto_18899_837.json` | 18898.85 | 837 | `start_guard=325`, `lookahead=70`, baseline scoring weights |
| `pareto_18742_798.json` | 18742.42 | 798 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043` plus `task_bonus=150` for `NQ08692/YT0294/NQ10422/NQ10421` |
| `pareto_18896_815.json` | 18896.39 | 815 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043/KB0040/KB0041/KB0044/KB0045/KB0046` to phase2 |
| `pareto_18907_823.json` | 18906.89 | 823 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043/KB0040/KB0041/KB0044/KB0045` to phase2 |
| `pareto_18932_825.json` | 18931.91 | 825 | `start_guard=325`, force `KB0040/KB0041/KB0044/KB0045/KB0046` to path `1` |
| `pareto_18932_826.json` | 18932.41 | 826 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043` to phase2, default paths |
| `pareto_18932_827.json` | 18932.41 | 827 | `start_guard=325`, force `KB0032/KB0037/KB0038/KB0039/KB0043` to path `1` |

Observed local search behavior:

| Experiment | Best observed | Notes |
| --- | --- | --- |
| Start-guard grid | `18898.85 / 837` | `start_guard=325`, `330`, and `335` converge to the same high-output schedule. |
| Wider lookahead near `sg330` | Below best | `lookahead=75..90` reduced output, despite sometimes lowering setup. |
| Higher weight / lower setup penalty | Below best | Relaxing setup or increasing weight disrupted the tail task packing. |
| Started bonus micro-search | Below best | Moving `score_started` away from `140` reduced output. |
| KB force-path sacrifice bundles | `18932.41 / 827` | Sacrificing selected low-yield late KB tasks onto path `1` frees enough bottleneck capacity to pull higher-value tasks inside the horizon. |

Net gain versus `pareto_18875_818.json`:

| Metric | Delta |
| --- | ---: |
| Completed weight | +23.70 |
| Completed tasks | +1 |
| Positive setup count | +19 |

Task-set difference:

| Direction | Count | Weight |
| --- | ---: | ---: |
| Newly completed by `pareto_18899_837.json` | 7 | 75.80 |
| No longer completed versus `pareto_18875_818.json` | 6 | 52.10 |

The gain mainly comes from pulling late `YT0295..YT0298` and `YT0399..YT0400` tasks inside the horizon.

Validator note:

`validate` now infers task path choices from the output file when `--force-path` is not provided, so force-path Pareto files can be checked directly with the normal validate command.

## 2026-04-25 Follow-up Probes

No new Pareto point exceeded the current high-output best `pareto_18932_826.json`, but the near-horizon bonus probe added a useful setup-under-800 point at `18742.42 / 798`.

| Probe | Best observed | Outcome |
| --- | --- | --- |
| Estimated-finish penalty grid | `18870.75 / 780` | Lower `score_est_final_per` can reduce setup, but loses too much completed weight. The default `0.01` remains best for high output. |
| Phase2 finish penalty grid | `18932.41 / 827` | Values from `0` to `0.03` converged to the same incumbent schedule. |
| Reference-solution path subsets | `18723.65 / 828` | Forcing the reference `YT/TJ/KB/BC` path deltas on top of the incumbent disrupted the current heuristic schedule. |
| Critical late-task bonus | `18689.16 / 833` | Directly boosting near-horizon late tasks pulled the search away from higher-value incumbent packing. |
| Defer-task sacrifice | `18932.41 / 826` | Deferring the same five KB tasks that previously needed force-path sacrifices preserved output and removed one positive setup. Wider KB defer sets produced lower-setup Pareto points at `18906.89 / 823` and `18896.39 / 815`. |
| Defer plus near-horizon bonus | `18742.42 / 798` | A bonus of `140..155` for `NQ08692/YT0294/NQ10422/NQ10421` consistently crossed below 800 setup, improving the previous 800-ish frontier even though it sacrifices high-output packing. |

Automation note:

`search_relaxed_frontier.py` now anchors at `start_guard=330`, samples the exposed finish-penalty knobs, and accepts `--defer-task`, so future automated searches start from the current high-output basin instead of the older `18875` basin.

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

Observed local search behavior:

| Experiment | Best observed | Notes |
| --- | --- | --- |
| Start-guard grid | `18898.85 / 837` | `start_guard=325`, `330`, and `335` converge to the same high-output schedule. |
| Wider lookahead near `sg330` | Below best | `lookahead=75..90` reduced output, despite sometimes lowering setup. |
| Higher weight / lower setup penalty | Below best | Relaxing setup or increasing weight disrupted the tail task packing. |
| Started bonus micro-search | Below best | Moving `score_started` away from `140` reduced output. |

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

Pareto frontier for the actual-scale instance at horizon `24480`.

Input instance:

- [实际规模输入数据.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/data/data1/实际规模输入数据.json:1)

Definition used:

- maximize `completed_weight_within_horizon`
- minimize `setup_count_positive`
- only keep complete and valid solutions with `error_count = 0`
- batch machines are treated with the relaxed infinite-capacity assumption used by the solver

Frontier solutions:

| setup_count_positive | completed_weight_within_horizon | solution |
| --- | ---: | --- |
| 508 | 16514.39 | [pareto_16514_508.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_16514_508.json:1) |
| 600 | 18221.74 | [pareto_18222_600.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18222_600.json:1) |
| 644 | 18296.62 | [pareto_18297_644.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18297_644.json:1) |
| 647 | 18628.82 | [pareto_18629_647.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18629_647.json:1) |
| 688 | 18639.81 | [pareto_18640_688.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18640_688.json:1) |
| 731 | 18644.03 | [pareto_18644_731.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18644_731.json:1) |
| 743 | 18648.39 | [pareto_18648_743.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18648_743.json:1) |
| 746 | 18658.47 | [pareto_18658_746.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18658_746.json:1) |
| 788 | 18685.36 | [pareto_18685_788.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18685_788.json:1) |
| 806 | 18723.73 | [pareto_18724_806.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18724_806.json:1) |
| 809 | 18847.15 | [pareto_18847_809.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18847_809.json:1) |
| 810 | 18865.95 | [pareto_18866_810.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18866_810.json:1) |
| 818 | 18875.15 | [pareto_18875_818.json](F:/huawei_fjsp_llm/huawei_fjsp_llm/outputs/pareto_24480/pareto_18875_818.json:1) |

Notes:

- This is the current non-dominated set found from the accumulated local search runs so far. It is a searched Pareto frontier, not a proof of the global frontier.
- High-setup experimental runs above `1000` setups were also tested, but in the current search they were dominated and did not enter the frontier.
- The `788 / 18685.36` point comes from the path-aware search after fixing a repair-stage setup-consistency bug in the solver.
- The `809 / 18847.15` point comes from forcing `YT0363` from path `1` to path `0`.
- The `810 / 18865.95` point comes from the five-task force-path bundle `YT0363=0, KB0032=1, KB0037=1, KB0038=1, KB0043=1`.
- When validating the two new force-path points, pass the same `--force-path` overrides used to generate them; validating them under the default path-selection cache will report a path-coverage mismatch.

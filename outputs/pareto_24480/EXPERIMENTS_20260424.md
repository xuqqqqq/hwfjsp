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
| `pareto_18767_768.json` | 18766.98 | 768 | Defer the low-tail bundle except `NP71371`; this dominates the earlier `18764 / 775` point |
| `pareto_18772_802.json` | 18771.72 | 802 | Defer the low-tail bundle except `NP71371/NP99641`; higher output than `18767 / 768` at a small setup increase |
| `pareto_18764_775.json` | 18763.98 | 775 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043` plus low-yield tail tasks `NP71371/NP99641/NP9387/TJ4511/TJ4703/ED52681/ED52682/NP90961/NP90962/ED52701/ED52702` |
| `pareto_18742_798.json` | 18742.42 | 798 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043` plus `task_bonus=150` for `NQ08692/YT0294/NQ10422/NQ10421` |
| `pareto_18896_815.json` | 18896.39 | 815 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043/KB0040/KB0041/KB0044/KB0045/KB0046` to phase2 |
| `pareto_18907_823.json` | 18906.89 | 823 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043/KB0040/KB0041/KB0044/KB0045` to phase2 |
| `pareto_18932_825.json` | 18931.91 | 825 | `start_guard=325`, force `KB0040/KB0041/KB0044/KB0045/KB0046` to path `1` |
| `pareto_18932_826.json` | 18932.41 | 826 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043` to phase2, default paths |
| `pareto_18932_827.json` | 18932.41 | 827 | `start_guard=325`, force `KB0032/KB0037/KB0038/KB0039/KB0043` to path `1` |
| `pareto_18943_835.json` | 18942.79 | 835 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043/NP99641/NQ02772`; trades `NQ02772` for `NQ0507`, lowering setup versus the high-output best |
| `pareto_18946_839.json` | 18945.59 | 839 | Defer `KB0032/KB0037/KB0038/KB0039/KB0043/NP99641`; trades a low-weight surface-treatment tail task for higher-value near-horizon NQ/YT completions |
| `pareto_18767_677.json` | 18766.98 | 677 | Add `--phase2-allow-unstarted` to the low-tail defer bundle; preserves the prior low-tail weight while removing 91 positive setups |
| `pareto_18784_691.json` | 18783.62 | 691 | Keep both `NP71371` and `NP99641` out of the low-tail defer bundle under the relaxed phase2 gate |
| `pareto_18943_733.json` | 18942.79 | 733 | Add `--phase2-allow-unstarted` to the surface setup bridge; same completed weight as `18943 / 835` with 102 fewer positive setups |
| `pareto_18946_716.json` | 18945.59 | 716 | Extend the machine-level tail repack with `NP8539`, `NQ1152`, and `NQ1144` operation-machine overrides; same completed weight as `18946 / 729` with 13 fewer positive setups |
| `pareto_18946_729.json` | 18945.59 | 729 | Add `--force-machine YT0294:8=4#拉弯矫` on top of the high-output phase2-gate point; same completed weight as `18946 / 737` with 8 fewer positive setups |
| `pareto_18946_737.json` | 18945.59 | 737 | Add `--phase2-allow-unstarted` to the high-output `NP99641` surface-tail sacrifice; same completed weight as `18946 / 839` with 102 fewer positive setups |

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

Tail-sacrifice probes improved the low-setup region to `18766.98 / 768` and added an adjacent `18771.72 / 802` tradeoff. A later surface-treatment tail probe also raised the high-output best to `18945.59 / 839`.

| Probe | Best observed | Outcome |
| --- | --- | --- |
| Estimated-finish penalty grid | `18870.75 / 780` | Lower `score_est_final_per` can reduce setup, but loses too much completed weight. The default `0.01` remains best for high output. |
| Phase2 finish penalty grid | `18932.41 / 827` | Values from `0` to `0.03` converged to the same incumbent schedule. |
| Reference-solution path subsets | `18723.65 / 828` | Forcing the reference `YT/TJ/KB/BC` path deltas on top of the incumbent disrupted the current heuristic schedule. |
| Critical late-task bonus | `18689.16 / 833` | Directly boosting near-horizon late tasks pulled the search away from higher-value incumbent packing. |
| Defer-task sacrifice | `18932.41 / 826` | Deferring the same five KB tasks that previously needed force-path sacrifices preserved output and removed one positive setup. Wider KB defer sets produced lower-setup Pareto points at `18906.89 / 823` and `18896.39 / 815`. |
| Defer plus near-horizon bonus | `18742.42 / 798` | A bonus of `140..155` for `NQ08692/YT0294/NQ10422/NQ10421` consistently crossed below 800 setup, improving the previous 800-ish frontier even though it sacrifices high-output packing. |
| Defer low-yield tail bundle | `18763.98 / 775` | Deferring low-weight tasks finishing near the horizon produced a cleaner low-setup tradeoff and dominates the `18742.42 / 798` intermediate point. |
| Low-tail bundle ablation | `18766.98 / 768`; `18771.72 / 802` | Keeping `NP71371` improves both weight and setup. Keeping both `NP71371` and `NP99641` buys a little more weight at setup `802`. |
| Surface-treatment tail sacrifice | `18945.59 / 839` | Deferring `NP99641` on top of the five-KB defer set pulls `NQ08711/YT0627/NQ08691/NQ08692` inside the horizon while losing `YT0736/ED5683/NP99641`, netting `+13.18` completed weight. |
| Surface setup bridge | `18942.79 / 835` | Adding `NQ02772` to the `NP99641` defer set swaps in `NQ0507` and removes four positive setup transitions at a small `-2.80` weight cost. |
| Phase2 unstarted candidate gate | `18945.59 / 737`; `18942.79 / 733`; `18783.62 / 691`; `18766.98 / 677` | Allowing phase2 to score unstarted tasks even when started candidates exist preserves the within-horizon task set for the high-output schedules but repacks the after-horizon completion tail with far fewer setup transitions. Zero-setup bonus variants under this mode produced incomplete solutions and were rejected. |
| Machine-level tail repack | `18945.59 / 716` | Forcing late `YT0294:8=4#拉弯矫`, `NP8539:3=4#冷轧机`, `NP8539:4=3#拉弯矫`, `NQ1152:10=3#重卷机`, and `NQ1144:5=3#重卷机` leaves the completed-within-horizon task set unchanged while lowering positive setups. Rejected extensions include `YT0294` to `4#纵剪`, `YT0295` final-machine overrides, `YT0312:4=3#重卷机`, and `ED5608:1=2#重卷机` because they reduced weight or produced incomplete schedules. |

Automation note:

`search_relaxed_frontier.py` now anchors at `start_guard=330`, samples the exposed finish-penalty knobs, and accepts `--defer-task`, so future automated searches start from the current high-output basin instead of the older `18875` basin.

The search script also accepts `--phase2-allow-unstarted` and `--force-machine`; both should be available for current frontier searches because phase2 repacking and targeted machine-level tail moves are now the strongest setup reducers at fixed high output.

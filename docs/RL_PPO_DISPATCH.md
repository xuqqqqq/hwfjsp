# RL PPO Dispatch Route

这版实现把现有的 `rl_relaxed_solver.py` 包装成了一个真正可训练的 phase-1 dispatch 环境，目标是先学会在 `max_output_horizon` 内做更好的非 batch 决策，再把完整修复交回启发式补全。

## 设计选择

- 环境只训练第一阶段 horizon-focused 决策。
- batch 前缀仍然走确定性规则，避免动作空间爆炸。
- 动作不是“全局任意工序”，而是当前 `lookahead` 内候选的 top-k。
- 策略网络直接对候选列表打分，天然适合变长候选集合。
- 训练目标优先对齐 `24480` 内完成产量，同时惩罚 setup 和明显拖后。

## 文件

- `scripts/rl_phase1_env.py`
  - 从现有调度器提取可 step 的 phase-1 环境
  - 支持子算例采样、候选特征构造、启发式基线动作
- `scripts/rl_train_ppo.py`
  - PPO 训练入口
  - 支持先做 heuristic behavior cloning warm start，再做 PPO 微调
  - 支持 curriculum task-limit 训练和每轮 imitation regularization
  - 支持 smoke 模式、小规模子集训练、checkpoint 落盘

## 依赖

- `torch`
- `numpy`
- 复用项目现有的 `ijson`

## 推荐起步命令

```powershell
python scripts/rl_train_ppo.py --smoke
```

更正式一点的训练：

```powershell
python scripts/rl_train_ppo.py `
  --task-limit 256 `
  --task-sampling mixed `
  --decision-budget 512 `
  --steps-per-update 1024 `
  --updates 10
```

如果想显式控制 warm start：

```powershell
python scripts/rl_train_ppo.py `
  --bc-steps 1024 `
  --bc-epochs 3 `
  --updates 10
```

如果想逐步放大任务规模：

```powershell
python scripts/rl_train_ppo.py `
  --curriculum-task-limits 64,128,256 `
  --imitation-steps-per-update 128 `
  --imitation-coef 0.05 `
  --updates 9
```

## 这版的边界

- 还没有把 phase-2 补全也纳入 RL。
- 还没有做路径选择策略网络。
- 当前奖励仍然是工程化 shaped reward，不是最终业务指标的唯一来源。
- 这版更像“能真正开始训练的底座”，不是已经打赢当前最强启发式的最终体。
- 目前更现实的目标是：先在子算例上稳定贴住 heuristic baseline，再逐步扩大到全实例。

Preserved historical baseline:

- Solution: `outputs/rl_relaxed_full_solution.json`
- Mirror: `outputs/rl_relaxed_full_solution_18524_767.json`
- Completed weight within horizon 24480: `18524.28`
- Positive setup transitions: `767`
- Fully scheduled tasks: `1751 / 1751`
- Validation errors: `0`

Current best higher-output candidate:

- Candidate: `outputs/rl_relaxed_candidate_18658_746.json`
- Completed weight within horizon 24480: `18658.47`
- Positive setup transitions: `746`
- Fully scheduled tasks: `1751 / 1751`
- Validation errors: `0`

Current best balanced candidate and default solver config:

- Candidate: `outputs/rl_relaxed_candidate_18640_688.json`
- Completed weight within horizon 24480: `18639.81`
- Positive setup transitions: `688`
- Fully scheduled tasks: `1751 / 1751`
- Validation errors: `0`

Legacy under-800 setup checkpoints:

- `outputs/rl_relaxed_candidate_18547_784.json` -> `18547.29 / 784`
- `outputs/rl_relaxed_candidate_18598_791.json` -> `18598.93 / 791`

Solver entrypoint:

- `python scripts/rl_relaxed_solver.py solve`

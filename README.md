# Simulation Baseline

## Requirements
- Python 3.9+
- numpy, matplotlib, pytest

## Run the simulation
From the repo root:

```bash
python -m src.run_sim
```

Optional knobs:
- `--num-cohorts N`: assign users to cohorts and run independent Beta-TS per cohort.
- `--segment-len K`: hold the same user fixed for K rounds (ablation).
- `--impressions-per-pull M`: aggregate M impressions per chosen arm to reduce variance.

This prints:
- random policy average click rate
- no-edit greedy policy average click rate
- Thompson sampling policy average click rate
- constrained oracle average click rate

It also prints:
- fraction of candidate edited arms rejected
- fraction of feasible edited arms that beat no-edit

And saves `click_rate.png` in the repo root.

## Run tests

```bash
pytest
```

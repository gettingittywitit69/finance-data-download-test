# Sharpe MC Study

Minimal, dependency-light Monte Carlo framework to compare naive vs dependence-robust Sharpe-ratio inference across a few stylized return processes.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run experiment
```bash
python -m src.run_mc --reps 20000 --seed 42 --out results/summary.csv
```
Progress prints once; defaults cover 24 cells (4 DGPs × 3 n-grid × 2 Sharpe levels) and four methods (naive, robust HAC, PSR, DSR). On a modern laptop the default run takes a few minutes; use `--max-workers` to cap processes or lower `--reps` for smoke runs. Add `--epsilon 0.01` to print the required rep count for a 1% half-width at 95% confidence (Bonferroni over all cells).

The CSV columns are: `dgp, n, S_true, method, bias, rmse, coverage_95, reject_rate_H0_S_le_0, se_ratio, psr_reject_rate, dsr_reject_rate`.

## Tests
```bash
pytest -q
```

## Notes
- DGPs: iid normal, iid t(df=5), AR(1)-t(phi=0.3), GARCH(1,1)-t(alpha=0.05, beta=0.90) simulated via `arch` with Student-t innovations.
- Methods: naive asymptotic, robust HAC/Newey-West, PSR baseline, DSR baseline.
- True Sharpe grid: {0.0, 0.5}; n grid: {120, 240, 1200}; default reps=20000; sigma=0.04 (monthly).

## Changelog
- Removed manual GARCH recursion in favor of `arch` simulation for speed and correctness.
- Dropped bandwidth-tag grid flags; HAC now uses the standard automatic Newey–West rule.
- Consolidated orchestration into a single process-pooled run path and added PSR/DSR baselines and replication calculator utilities.

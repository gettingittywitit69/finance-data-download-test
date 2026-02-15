from __future__ import annotations

import argparse

from . import sharpe_mc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo grid for Sharpe-ratio inference")
    parser.add_argument("--reps", type=int, default=sharpe_mc.DEFAULT_REPS, help="replications per cell")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument("--out", type=str, default="results/summary.csv", help="output CSV path")
    parser.add_argument("--max-workers", type=int, default=None, help="process workers (default: cpu count)")
    parser.add_argument("--no-sanity", action="store_true", help="skip runtime sanity checks")
    parser.add_argument("--n-trials", type=int, default=sharpe_mc.DEFAULT_N_TRIALS, help="trial count for DSR baseline")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="optional half-width target; print required reps at alpha=0.05",
    )
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()
    cell_count = len(sharpe_mc.DGP_LIST) * len(sharpe_mc.N_GRID) * len(sharpe_mc.S_TRUE_GRID)

    if args.epsilon is not None:
        req = sharpe_mc.required_reps(alpha=0.05, epsilon=args.epsilon, p=0.5, m_cells=cell_count)
        print(f"Required reps for epsilon={args.epsilon:.4f} at alpha=0.05 over {cell_count} cells: {req}")

    print(
        f"Running MC grid: reps={args.reps}, seed={args.seed}, workers={args.max_workers or 'auto'}, "
        f"cells={cell_count}"
    )
    rows = sharpe_mc.run_experiment(
        reps=args.reps,
        seed=args.seed,
        max_workers=args.max_workers,
        n_trials=args.n_trials,
        run_sanity=not args.no_sanity,
    )
    sharpe_mc.write_summary_csv(rows, args.out)
    print(f"Done. Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    run_cli()

"""ACO improvement comparison.

Runs the same 200-pickrun sample through six ACO variants to isolate the
contribution of each improvement added in Task 1:

  baseline    — original port (no symmetric deposit, no elitism, no 2-opt)
  +sym        — 1a only: symmetric pheromone deposit
  +elite      — 1b only: elitism (global-best reinforcement)
  +2opt       — 1c only: per-ant 2-opt before deposit
  +sym+elite  — 1a + 1b combined
  all         — all three (the current default)

All non-ACO solvers (nn, 2opt, or_opt, aisle_nn, bucketed, sa, mst) are
included as reference points so the ACO variants are benchmarked against the
full solver field (the same set run_demo.py compares).

Usage
-----
    python compare_aco.py
"""
from __future__ import annotations

import time

from data_gen import generate
from graph import WarehouseGraph
from tsp import route_all_pickruns


def _pct(baseline: float, val: float) -> str:
    if baseline == 0:
        return "  n/a"
    return f"{100.0 * (baseline - val) / baseline:+.1f}%"


def main() -> None:
    print("Generating data...")
    ds = generate(n_aisles=10, n_positions=20, n_items=300,
                  n_pickruns=500, n_days=1, seed=42)
    graph = WarehouseGraph(ds.locations)

    VARIANTS = [
        # (label,          solver,  extra kwargs)
        # All non-ACO solvers as reference points (same field as run_demo.py),
        # so the ACO variants are benchmarked against the whole solver set.
        ("NN (reference)",       "nn",       {}),
        ("2-opt (reference)",    "2opt",     {}),
        ("or-opt (reference)",   "or_opt",   {}),
        ("aisle-NN (reference)", "aisle_nn", {}),
        ("bucketed (reference)", "bucketed", {}),
        ("SA (reference)",       "sa",       {}),
        ("MST (reference)",      "mst",      {}),
        ("ACO baseline",    "aco",  dict(symmetric_deposit=False, elitism=False, per_ant_2opt=False)),
        ("ACO +sym",        "aco",  dict(symmetric_deposit=True,  elitism=False, per_ant_2opt=False)),
        ("ACO +elite",      "aco",  dict(symmetric_deposit=False, elitism=True,  per_ant_2opt=False)),
        ("ACO +2opt",       "aco",  dict(symmetric_deposit=False, elitism=False, per_ant_2opt=True)),
        ("ACO +sym+elite",  "aco",  dict(symmetric_deposit=True,  elitism=True,  per_ant_2opt=False)),
        ("ACO all three",   "aco",  dict(symmetric_deposit=True,  elitism=True,  per_ant_2opt=True)),
    ]

    print(f"\n{'Variant':<24}  {'Avg dist/run':>12}  {'vs 2opt':>7}  {'vs baseline':>11}  {'Time':>7}")
    print(f"{'─'*24}  {'─'*12}  {'─'*7}  {'─'*11}  {'─'*7}")

    ref_2opt: float = 0.0
    aco_baseline: float = 0.0

    for label, solver, kw in VARIANTS:
        t0 = time.time()
        try:
            rs = route_all_pickruns(
                ds.transactions, graph, ds.items,
                max_pickruns=200, solver=solver, **kw,
            )
        except ImportError as exc:
            print(f"{label:<24}  skipped ({exc})")
            continue
        elapsed  = time.time() - t0
        avg_dist = sum(r.total_dist_m for r in rs) / max(1, len(rs))

        vs_2opt     = _pct(ref_2opt, avg_dist)     if ref_2opt     else "  base"
        vs_baseline = _pct(aco_baseline, avg_dist) if aco_baseline else "      —"

        print(f"{label:<24}  {avg_dist:>10.1f} m  {vs_2opt:>7}  {vs_baseline:>11}  {elapsed:>5.2f}s")

        if label == "2-opt (reference)":
            ref_2opt = avg_dist
        if label == "ACO baseline":
            aco_baseline = avg_dist


if __name__ == "__main__":
    main()

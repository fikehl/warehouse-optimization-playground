"""Task 2a — Threshold sensitivity: makespan vs reroute_on_wait_s vs n_pickers.

Usage
-----
    python task2a_threshold_sensitivity.py

Output
------
    task2a_threshold_sensitivity.png  — two-panel matplotlib figure
"""
from __future__ import annotations

import sys

import numpy as np

from data_gen import generate
from graph import WarehouseGraph
from tsp import route_all_pickruns
from des import run_des

# ---------------------------------------------------------------------------
# Setup (same synthetic dataset as run_demo.py)
# ---------------------------------------------------------------------------
print("Generating data and building graph...")
ds    = generate(n_aisles=10, n_positions=20, n_items=300, n_pickruns=500, n_days=1, seed=42)
graph = WarehouseGraph(ds.locations)
routes = route_all_pickruns(ds.transactions, graph, ds.items, solver="nn")

# IMPORTANT: synchronise releases so the system is contention-bound.
#
# The raw transaction timestamps spread pickrun releases over a ~16h day, which
# makes the makespan release-bound: the last pickrun simply can't start until
# it is released, so contention (and therefore rerouting) has negligible effect
# and a threshold sweep produces a flat line.  Rerouting only matters when many
# pickers compete for one-way aisles at the same time, so for this study we
# release every pickrun at t=0.  Comment out this line to see the release-bound
# (no-effect) regime instead.
routes = [r._replace(release_s=0.0) for r in routes]
print(f"  {len(routes)} pickruns routed (releases synchronised to t=0 for contention).")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
THRESHOLDS   = [5, 10, 20, 30, 60]   # reroute_on_wait_s values
PICKER_COUNTS = [2, 3, 5]

# results[(n_p, thresh)] = {"makespan_s": ..., "n_reroutes": ...}
results: dict[tuple[int, int | str], dict] = {}

for n_p in PICKER_COUNTS:
    print(f"\nn_pickers={n_p}")
    base = run_des(routes, graph, n_pickers=n_p)
    results[(n_p, "baseline")] = base
    print(f"  baseline        makespan={base['makespan_s']/60:.1f} min")

    for thresh in THRESHOLDS:
        stats = run_des(routes, graph, n_pickers=n_p,
                        reroute_on_wait_s=float(thresh), reroute_solver="nn")
        results[(n_p, thresh)] = stats
        pct = 100.0 * (base["makespan_s"] - stats["makespan_s"]) / base["makespan_s"]
        print(f"  thresh={thresh:>3}s  makespan={stats['makespan_s']/60:.1f} min "
              f"  ({pct:+.1f}%)  reroutes={stats['n_reroutes']}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed — skipping plot (pip install matplotlib).")
    sys.exit(0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Task 2a — Threshold sensitivity: reroute_on_wait_s  (solver=nn)", fontsize=13)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# --- Panel 1: absolute makespan ---
ax = axes[0]
for n_p, color in zip(PICKER_COUNTS, colors):
    ys = [results[(n_p, t)]["makespan_s"] / 60 for t in THRESHOLDS]
    ax.plot(THRESHOLDS, ys, marker="o", color=color, label=f"{n_p} pickers")
    base_min = results[(n_p, "baseline")]["makespan_s"] / 60
    ax.axhline(base_min, linestyle="--", color=color, alpha=0.45,
               label=f"baseline ({n_p}p)")
ax.set_xlabel("reroute_on_wait_s (s)")
ax.set_ylabel("Makespan (min)")
ax.set_title("Absolute makespan")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: % improvement over baseline ---
ax = axes[1]
for n_p, color in zip(PICKER_COUNTS, colors):
    base_ms = results[(n_p, "baseline")]["makespan_s"]
    ys = [100.0 * (base_ms - results[(n_p, t)]["makespan_s"]) / base_ms
          for t in THRESHOLDS]
    ax.plot(THRESHOLDS, ys, marker="o", color=color, label=f"{n_p} pickers")
ax.axhline(0, color="black", linestyle="--", alpha=0.35, linewidth=0.8)
ax.set_xlabel("reroute_on_wait_s (s)")
ax.set_ylabel("Makespan improvement vs baseline (%)")
ax.set_title("% improvement over no-rerouting baseline")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: reroute count ---
ax = axes[2]
x_idx = np.arange(len(THRESHOLDS))
bar_w = 0.25
for k, (n_p, color) in enumerate(zip(PICKER_COUNTS, colors)):
    ys = [results[(n_p, t)]["n_reroutes"] for t in THRESHOLDS]
    ax.bar(x_idx + k * bar_w, ys, bar_w, color=color, alpha=0.8, label=f"{n_p} pickers")
ax.set_xticks(x_idx + bar_w)
ax.set_xticklabels([str(t) for t in THRESHOLDS])
ax.set_xlabel("reroute_on_wait_s (s)")
ax.set_ylabel("Total reroutes triggered")
ax.set_title("Rerouting frequency")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = "task2a_threshold_sensitivity.png"
plt.savefig(out, dpi=130)
print(f"\nSaved {out}")

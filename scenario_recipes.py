"""Scenario recipes — run every README recipe against the task it stresses most.

The README's "Scenario recipes" section gives seven generator variations plus a
zone-crossing pointer.  This script runs each one with the experiment it is
most relevant to:

  R1  Busy single shift     -> DES regime check (release-bound vs contention)
  R2  Large warehouse       -> Task 1 solver comparison at scale + DES
  R3  Small dense warehouse -> Task 1 ACO headroom + Task 2 rerouting payoff
  R4  Heavy-goods warehouse -> weight-ordering (segmentation) effects
  R5  Highly skewed demand  -> demand-profile analysis (see note in output)
  R6  Multi-day with trend  -> picks_trend fast/slow mover analysis
  R7  Cold-chain heavy      -> Task 4 cold-last cost amplification
  R8  Zone-crossing         -> covered by task4_cold_chain.py (4d) — pointer only

Usage
-----
    python scenario_recipes.py
"""
from __future__ import annotations

import time

import numpy as np

from data_gen import generate
from graph import WarehouseGraph
from tsp import route_all_pickruns, HEAVY_KG, FRAGILE_KG
from des import run_des


def _avg_dist(rs) -> float:
    return sum(r.total_dist_m for r in rs) / max(1, len(rs))


def _avg_fixes(rs) -> float:
    return sum(r.ordering_fixes for r in rs) / max(1, len(rs))


def _sync(rs):
    return [r._replace(release_s=0.0) for r in rs]


# ===========================================================================
# R1 — Busy single shift  (all 500 orders on one day)
# ===========================================================================
print("=== R1 — Busy single shift: generate(n_pickruns=500, n_days=1) ===")
print("  This recipe is already the baseline of every task script.  The key")
print("  property it exposes is the DES regime:")
ds = generate(n_pickruns=500, n_days=1, seed=42)
g  = WarehouseGraph(ds.locations)
routes = route_all_pickruns(ds.transactions, g, ds.items, solver="nn")
s_stag = run_des(routes, g, n_pickers=3)
s_sync = run_des(_sync(routes), g, n_pickers=3)
print(f"  staggered releases (raw timestamps): makespan={s_stag['makespan_s']/60:>7.1f} min"
      f"  avg aisle wait={s_stag['avg_one_way_wait_s']:.2f} s")
print(f"  synchronised releases (t=0)        : makespan={s_sync['makespan_s']/60:>7.1f} min"
      f"  avg aisle wait={s_sync['avg_one_way_wait_s']:.2f} s")
print("  -> with raw timestamps the makespan is RELEASE-BOUND (~16 h window);")
print("     synchronising to t=0 makes it contention-bound — which is why every")
print("     task experiment syncs releases before measuring anything.")

# ===========================================================================
# R2 — Large warehouse  (20 aisles x 40 positions, 800 SKUs, 1000 orders)
# ===========================================================================
print("\n=== R2 — Large warehouse: generate(n_aisles=20, n_positions=40, "
      "n_items=800, n_pickruns=1000) ===")
ds2 = generate(n_aisles=20, n_positions=40, n_items=800, n_pickruns=1000,
               n_days=1, seed=42)
g2  = WarehouseGraph(ds2.locations)
print(f"  {len(ds2.locations)} locations, one-way aisles: {len(g2.oneway_xs())}")
print(f"  {'Solver':<10}  {'avg dist/run':>12}  {'time':>7}")
for solver in ["nn", "2opt", "or_opt", "aisle_nn", "aco"]:
    t0 = time.time()
    rs = route_all_pickruns(ds2.transactions, g2, ds2.items,
                            max_pickruns=200, solver=solver)
    print(f"  {solver:<10}  {_avg_dist(rs):>10.1f} m  {time.time()-t0:>6.1f}s")
routes2 = _sync(route_all_pickruns(ds2.transactions, g2, ds2.items, solver="nn"))
for n_p in [3, 6]:
    s = run_des(routes2, g2, n_pickers=n_p)
    print(f"  DES nn routes, {n_p} pickers: makespan={s['makespan_s']/3600:.2f} h  "
          f"avg aisle wait={s['avg_one_way_wait_s']:.2f} s")
print("  -> longer aisles raise per-run distance; contention per aisle drops")
print("     (10 one-way aisles share the same picker count).")

# ===========================================================================
# R3 — Small dense warehouse  (4 aisles x 30 positions, long pickruns)
# ===========================================================================
print("\n=== R3 — Small dense warehouse: generate(n_aisles=4, n_positions=30, "
      "n_items=100, max_lines_per_run=15) ===")
ds3 = generate(n_aisles=4, n_positions=30, n_items=100, max_lines_per_run=15,
               n_pickruns=500, n_days=1, seed=42)
g3  = WarehouseGraph(ds3.locations)
rs_nn = route_all_pickruns(ds3.transactions, g3, ds3.items, solver="nn")
print(f"  picks/run: mean {np.mean([r.n_picks for r in rs_nn]):.1f}, "
      f"one-way aisles: {sorted(g3.oneway_xs())}")

# Task 1 angle: ACO headroom grows with segment size
print(f"\n  Task 1 angle — ACO headroom on long pickruns (120-run sample):")
print(f"  {'Variant':<14}  {'avg dist/run':>12}  {'time':>7}")
for label, kw in [
    ("nn",        dict(solver="nn")),
    ("2opt",      dict(solver="2opt")),
    ("ACO base",  dict(solver="aco", symmetric_deposit=False, elitism=False, per_ant_2opt=False)),
    ("ACO all",   dict(solver="aco", symmetric_deposit=True,  elitism=True,  per_ant_2opt=True)),
]:
    t0 = time.time()
    rs = route_all_pickruns(ds3.transactions, g3, ds3.items, max_pickruns=120, **kw)
    print(f"  {label:<14}  {_avg_dist(rs):>10.1f} m  {time.time()-t0:>6.1f}s")

# Task 2 angle: rerouting payoff under high contention
print(f"\n  Task 2 angle — rerouting payoff under high contention (synced releases):")
routes3 = _sync(rs_nn)
print(f"  {'pickers':>7}  {'baseline':>9}  {'thr=5s':>8}  {'thr=10s':>8}  {'thr=30s':>8}  (makespan min / % vs base)")
for n_p in [3, 5]:
    base = run_des(routes3, g3, n_pickers=n_p)["makespan_s"]
    cells = []
    for thr in [5.0, 10.0, 30.0]:
        ms = run_des(routes3, g3, n_pickers=n_p, reroute_on_wait_s=thr)["makespan_s"]
        cells.append(f"{100*(base-ms)/base:+.1f}%")
    print(f"  {n_p:>7}  {base/60:>7.1f}m  {cells[0]:>8}  {cells[1]:>8}  {cells[2]:>8}")

# ===========================================================================
# R4 — Heavy-goods warehouse  (most SKUs 200–2000 kg)
# ===========================================================================
print("\n=== R4 — Heavy-goods warehouse: weights -> uniform(200, 2000) ===")
ds4 = generate(seed=1, n_days=1)
g4  = WarehouseGraph(ds4.locations)
rs_before = route_all_pickruns(ds4.transactions, g4, ds4.items, solver="nn")
w_before  = ds4.items["weight"]
ds4.items["weight"] = np.random.default_rng(1).uniform(200, 2000, len(ds4.items)).round(1)
w_after   = ds4.items["weight"]
rs_after  = route_all_pickruns(ds4.transactions, g4, ds4.items, solver="nn")


def _cls_share(w) -> str:
    return (f"heavy={100*(w >= HEAVY_KG).mean():.0f}%  "
            f"normal={100*((w > FRAGILE_KG) & (w < HEAVY_KG)).mean():.0f}%  "
            f"fragile={100*(w <= FRAGILE_KG).mean():.0f}%")


print(f"  default weights : {_cls_share(w_before)}")
print(f"  heavy-goods     : {_cls_share(w_after)}")
print(f"  {'':<16}  {'avg dist/run':>12}  {'avg ordering_fixes':>19}")
print(f"  {'default':<16}  {_avg_dist(rs_before):>10.1f} m  {_avg_fixes(rs_before):>19.2f}")
print(f"  {'heavy-goods':<16}  {_avg_dist(rs_after):>10.1f} m  {_avg_fixes(rs_after):>19.2f}")
print("  -> same transactions, only the weight CLASSIFICATION changes.  Class")
print("     mix shifts how many heavy/normal/fragile segments each run splits")
print("     into, which moves both route distance and raw-order violations.")

# ===========================================================================
# R5 — Highly skewed demand  (Pareto shape 1.2 -> 0.5)
# ===========================================================================
print("\n=== R5 — Highly skewed demand: Pareto shape 1.2 -> 0.5 ===")
rng = np.random.default_rng(42)
for shape in [1.2, 0.5]:
    p60 = (np.random.default_rng(42).pareto(shape, 300) * 8).astype(int) + 1
    srt = np.sort(p60)[::-1]
    top5 = srt[: max(1, len(srt) // 20)].sum() / srt.sum()
    top20 = srt[: max(1, len(srt) // 5)].sum() / srt.sum()
    print(f"  shape={shape}: top-5% SKUs = {100*top5:.0f}% of picks, "
          f"top-20% = {100*top20:.0f}%")
print("  NOTE: this recipe only changes the items table.  The transaction")
print("  generator draws pick items UNIFORMLY from inventory (data_gen.py:")
print("  rng.choice(item_arr)), ignoring picks_30/60 — so demand skew does NOT")
print("  change routing or DES results in the current generator.  Making")
print("  transactions demand-weighted would be the natural extension.")

# ===========================================================================
# R6 — Multi-day demand with trend
# ===========================================================================
print("\n=== R6 — Multi-day demand with trend: generate(n_pickruns=2000, "
      "n_days=90, seed=7) ===")
ds6 = generate(n_pickruns=2000, n_days=90, seed=7)
tr  = ds6.items["picks_trend"]
print(f"  items: {len(ds6.items)}, avg picks_trend={tr.mean():.2f} "
      f"(median {tr.median():.2f})")
print(f"  README thresholds: accelerating (>1.3): {(tr > 1.3).sum()} SKUs, "
      f"declining (<0.7): {(tr < 0.7).sum()} SKUs")
print(f"  rescaled (x0.5)  : accelerating (>0.65): {(tr > 0.65).sum()} SKUs, "
      f"declining (<0.35): {(tr < 0.35).sum()} SKUs")
print("  NOTE: picks_trend = picks_30/picks_60, and picks_30 covers HALF the")
print("  window of picks_60 — so steady demand gives ~0.5, not 1.0.  The README")
print("  thresholds (1.3 / 0.7) assume a normalised ratio and misclassify almost")
print("  everything as declining; halve them (0.65 / 0.35) for sensible flags.")
print("  -> a data-profiling recipe: no routing/DES dimension (and per R5,")
print("     transactions are not demand-weighted anyway).")

# ===========================================================================
# R7 — Cold-chain heavy  (80% of slots CHILLER/FREEZER)
# ===========================================================================
print("\n=== R7 — Cold-chain heavy: 80% of LOCATIONS reassigned to cold zones ===")
print("  NOTE: the README snippet reassigns ds.items['zone_type_1'], but routing")
print("  reads zones from ds.LOCATIONS (loc_zone lookup in route_pickrun), and")
print("  post-generation item zones don't re-slot inventory — so the snippet as")
print("  written is a no-op for cold_last.  Applying it to locations instead:")
ds7 = generate(seed=3, n_days=1)
ds7.locations["zone_type_1"] = np.where(
    np.random.default_rng(3).random(len(ds7.locations)) < 0.8,
    np.random.default_rng(4).choice(["CHILLER", "FREEZER"], len(ds7.locations)),
    "AMBIENT",
)
g7 = WarehouseGraph(ds7.locations)
base7 = route_all_pickruns(ds7.transactions, g7, ds7.items, solver="nn")
cold7 = route_all_pickruns(ds7.transactions, g7, ds7.items,
                           locations_df=ds7.locations, solver="nn", cold_last=True)
print(f"  {'':<22}  {'avg dist/run':>12}  {'avg ordering_fixes':>19}")
print(f"  {'nn (no constraint)':<22}  {_avg_dist(base7):>10.1f} m  {_avg_fixes(base7):>19.2f}")
print(f"  {'nn cold-last':<22}  {_avg_dist(cold7):>10.1f} m  {_avg_fixes(cold7):>19.2f}")
print(f"  distance penalty: {100*(_avg_dist(cold7)-_avg_dist(base7))/_avg_dist(base7):+.1f}%")
print("  -> 80% cold AND interspersed (random per location): the cold-last")
print("     constraint now binds nearly every pick, so violations and the")
print("     distance penalty are both far larger than the default layout")
print("     (compare task4: grouped -2.2%, interspersed 70/20/10 +10.6%).")

# ===========================================================================
# R8 — Zone-crossing experiments
# ===========================================================================
print("\n=== R8 — Zone-crossing experiments ===")
print("  Implemented as Task 4d: run_des(zone_lookup=..., zone_cross_penalty_s=s).")
print("  See task4_cold_chain.py and task4_cold-chain.txt for results.")

print("\nDone.")

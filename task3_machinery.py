"""Task 3 — Mixed machinery fleet.

Covers all four sub-tasks:

  3a  Profile comparison        — HUMAN vs REACH_TRUCK vs COUNTERBALANCE.
  3b  Mixed fleet               — run_des(machine_profiles=[...]) with
                                  capability-based pickrun eligibility.
  3c  Electric pallet jack      — at what picker count does it match HUMAN?
  3d  Zone-restricted routing   — route_pickrun(can_use_oneway=False) vs the
                                  DES detour-penalty model.

Usage
-----
    python task3_machinery.py

Note on makespan
----------------
The raw transaction timestamps spread pickrun releases over a ~16 h day, which
makes makespan release-bound and hides the machine differences (every fleet
finishes at ~end-of-day).  As in task2a, we synchronise releases to t=0 so the
system is throughput/contention-bound and the machine profile actually drives
the makespan.  Set SYNC_RELEASES = False to see the release-bound regime.
"""
from __future__ import annotations

from collections import Counter

from data_gen import generate
from graph import WarehouseGraph
from tsp import route_all_pickruns, route_all_pickruns_by_class
from des import run_des, HUMAN, REACH_TRUCK, COUNTERBALANCE, PALLET_JACK

SYNC_RELEASES = True

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("Generating data and building graph...")
ds    = generate(n_aisles=10, n_positions=20, n_items=300, n_pickruns=500, n_days=1, seed=42)
graph = WarehouseGraph(ds.locations)
routes = route_all_pickruns(ds.transactions, graph, ds.items, solver="nn")
if SYNC_RELEASES:
    routes = [r._replace(release_s=0.0) for r in routes]
print(f"  {len(routes)} pickruns routed"
      f"{' (releases synchronised to t=0)' if SYNC_RELEASES else ''}.")


def _row(label: str, s: dict) -> None:
    print(f"  {label:<26}  {s['makespan_s']/60:>9.1f}  {s['avg_pickrun_time_s']:>9.1f}  "
          f"{s['avg_one_way_wait_s']:>9.2f}")


# ---------------------------------------------------------------------------
# 3a — Profile comparison (3 pickers each)
# ---------------------------------------------------------------------------
print("\n=== 3a — Profile comparison (3 pickers) ===")
print(f"  {'Machine':<26}  {'Makespan':>9}  {'Avg/run':>9}  {'AisleWait':>9}")
for p in [HUMAN, REACH_TRUCK, COUNTERBALANCE]:
    _row(p.name, run_des(routes, graph, n_pickers=3, machine_profile=p))
print("  (REACH_TRUCK usually wins both; COUNTERBALANCE shows 0 aisle-wait")
print("   because narrow_ok=False -> it never enters/queues for one-way aisles.)")

# ---------------------------------------------------------------------------
# 3b — Mixed fleet with PER-ITEM-CLASS routing (option C)
# ---------------------------------------------------------------------------
print("\n=== 3b — Mixed fleet, per-item-class routing ===")

# Split every order by the machine its items need: each item's class becomes a
# separate depot->picks->depot sub-route eligible only for its machine class,
# so one order can be served by up to three machines that consolidate at depot.
#
# NB: the default weight proxy (fragile<=50kg -> high_reach) finds nothing on
# this dataset — the slotting engine drops all light goods, so every slotted
# item is >340kg.  To exercise all three machine types we instead split the
# *actual* slotted weight range into terciles (heavy band -> counterbalance,
# mid band -> reach truck as a high-rack proxy, low band -> human).
import numpy as np
item_weight = dict(zip(ds.items["item"], ds.items["weight"]))
picked_w    = ds.transactions["item"].map(item_weight).dropna()
lo, hi      = np.quantile(picked_w, [1 / 3, 2 / 3])


def _band_class(item: str) -> str:
    w = item_weight.get(item, 0.0)
    if w >= hi:
        return "heavy"        # -> counterbalance
    if w >= lo:
        return "high_reach"   # -> reach_truck (high-rack proxy)
    return "normal"           # -> human


split_routes, eligibility = route_all_pickruns_by_class(
    ds.transactions, graph, ds.items, solver="nn", item_class=_band_class,
)
if SYNC_RELEASES:
    split_routes = [r._replace(release_s=0.0) for r in split_routes]

cls_for = {"counterbalance": "heavy", "reach_truck": "high_reach", "human": "normal"}
counts  = Counter(cls_for[next(iter(e))] for e in eligibility)
print(f"  {len(routes)} orders -> {len(split_routes)} class sub-routes "
      f"(heavy={counts['heavy']}, high_reach={counts['high_reach']}, normal={counts['normal']})")
print("  NOTE: 'Avg/run' below is per class sub-trip, not per whole order.")

print(f"  {'Fleet (H=human R=reach C=counter)':<34}  {'Makespan':>9}  {'Avg/trip':>9}")

# Each fleet must cover all three classes or run_des raises (guard).  With one
# machine per class the makespan = the single busiest machine's workload; the
# human is slowest (1.5 m/s vs reach 2.5, counter 3.5), so the normal class is
# the bottleneck.  Watch which additions actually move the makespan.
for label, fleet in [
    ("1H 1R 1C  (one each)",  [HUMAN, REACH_TRUCK, COUNTERBALANCE]),
    ("1H 1R 2C  (+counter)",  [HUMAN, REACH_TRUCK, COUNTERBALANCE, COUNTERBALANCE]),
    ("2H 1R 1C  (+human)",    [HUMAN, HUMAN, REACH_TRUCK, COUNTERBALANCE]),
    ("3H 2R 2C  (balanced)",  [HUMAN, HUMAN, HUMAN, REACH_TRUCK, REACH_TRUCK, COUNTERBALANCE, COUNTERBALANCE]),
]:
    s = run_des(split_routes, graph, machine_profiles=fleet,
                pickrun_eligibility=eligibility)
    print(f"  {label:<34}  {s['makespan_s']/60:>9.1f}  {s['avg_pickrun_time_s']:>9.1f}")
print("  -> +counter does nothing (heavy isn't the bottleneck); +human helps")
print("     because the slow human serves the normal class.  Balance the fleet")
print("     to the per-class workload, not by raw headcount.")

# ---------------------------------------------------------------------------
# 3c — Electric pallet jack vs HUMAN baseline
# ---------------------------------------------------------------------------
print("\n=== 3c — Electric pallet jack vs HUMAN ===")
jack = PALLET_JACK
print(f"  {'Pickers':>7}  {'HUMAN(min)':>11}  {'jack(min)':>11}  {'jack/HUMAN':>10}")
for n in [1, 2, 3, 4, 5, 6, 8]:
    h = run_des(routes, graph, n_pickers=n, machine_profile=HUMAN)["makespan_s"] / 60
    j = run_des(routes, graph, n_pickers=n, machine_profile=jack)["makespan_s"] / 60
    print(f"  {n:>7}  {h:>11.1f}  {j:>11.1f}  {j/h:>9.2f}x")
print("  Find the jack picker count whose makespan first beats HUMAN@3 above.")

# ---------------------------------------------------------------------------
# 3d — Zone-restricted routing: re-route vs detour-penalty
# ---------------------------------------------------------------------------
print("\n=== 3d — Wide-machine routing: re-route vs detour-penalty ===")

routes_oneway = route_all_pickruns(ds.transactions, graph, ds.items, solver="nn")
routes_bidir  = route_all_pickruns(ds.transactions, graph, ds.items, solver="nn",
                                   can_use_oneway=False)

oneway_xs = graph.oneway_xs()


def detour_effective_dist(rr) -> float:
    """Replicate the DES counterbalance cost: 2.2x on every one-way-aisle leg."""
    tiles = rr.route_tiles
    total = 0.0
    for a, b in zip(tiles, tiles[1:]):
        d = graph.distance(a, b)
        if graph.tile_x(b) in oneway_xs:
            d *= COUNTERBALANCE.detour_factor
        total += d
    return total


n = len(routes_oneway)
avg_oneway  = sum(r.total_dist_m for r in routes_oneway) / n
avg_bidir   = sum(r.total_dist_m for r in routes_bidir) / n
avg_penalty = sum(detour_effective_dist(r) for r in routes_oneway) / n

print(f"  {'Approach':<40}  {'Avg dist/run':>12}")
print(f"  {'one-way-respecting route (HUMAN baseline)':<40}  {avg_oneway:>10.1f} m")
print(f"  {'detour-penalty model (DES, 2.2x one-way)':<40}  {avg_penalty:>10.1f} m")
print(f"  {'re-routed on bidirectional graph (3d)':<40}  {avg_bidir:>10.1f} m")
print(f"\n  detour-penalty is {100*(avg_penalty-avg_oneway)/avg_oneway:+.1f}% vs baseline")
print(f"  bidirectional re-route is {100*(avg_bidir-avg_oneway)/avg_oneway:+.1f}% vs baseline")
print("  -> the flat 2.2x penalty assumes far more travel than an explicit")
print("     re-route, which (freed from one-way detours) is near or below baseline.")

print("\nDone.")

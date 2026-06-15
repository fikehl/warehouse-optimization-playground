"""Task 4 — Cold-chain constraints.

Covers all four sub-tasks:

  4a  Raw-order violations    — ordering_fixes with vs without cold_last.
  4b  Cost of the constraint  — cold-last distance penalty for a zone-grouped
                                warehouse vs an interspersed-zone one.
  4c  Dwell-time model        — DES cold-item dwell time (cold_last vs not).
  4d  Zone-crossing penalty   — does an env.timeout per zone change move which
                                solver wins?

Usage
-----
    python task4_cold_chain.py
"""
from __future__ import annotations

from data_gen import generate
from graph import WarehouseGraph
from tsp import route_all_pickruns
from des import run_des

SYNC_RELEASES = True   # contention/throughput-bound makespan (see task2/task3 notes)


def _tile_zone(graph: WarehouseGraph, locations) -> dict[str, str]:
    """Map each pick tile to its zone_type_1 (depot tiles are simply absent)."""
    return {graph.loc_to_tile(loc): z
            for loc, z in zip(locations["location_id"], locations["zone_type_1"])}


def _avg_dist(routes) -> float:
    return sum(r.total_dist_m for r in routes) / max(1, len(routes))


def _avg_fixes(routes) -> float:
    return sum(r.ordering_fixes for r in routes) / max(1, len(routes))


# ===========================================================================
# 4a — How often does the raw order violate cold-chain?
# ===========================================================================
print("=== 4a — Raw-order cold-chain violations (ordering_fixes) ===")
ds    = generate(n_aisles=10, n_positions=20, n_items=300, n_pickruns=500, n_days=1, seed=42)
graph = WarehouseGraph(ds.locations)

# Mirror run_demo.py's full solver comparison (200-pickrun sample).  ordering_fixes
# is computed from the RAW transaction order (before any solver runs), so it is
# identical for every solver — the table below makes that invariance explicit.
SOLVERS = ["nn", "2opt", "or_opt", "aisle_nn", "bucketed", "sa", "mst", "aco"]
print(f"  {'Solver':<10}  {'fixes (weight only)':>20}  {'fixes (+cold-chain)':>20}  {'extra':>7}")
for solver in SOLVERS:
    try:
        base = route_all_pickruns(ds.transactions, graph, ds.items,
                                  max_pickruns=200, solver=solver)
        cold = route_all_pickruns(ds.transactions, graph, ds.items,
                                  locations_df=ds.locations, max_pickruns=200,
                                  solver=solver, cold_last=True)
    except ImportError as exc:
        print(f"  {solver:<10}  skipped ({exc})")
        continue
    fb, fc = _avg_fixes(base), _avg_fixes(cold)
    print(f"  {solver:<10}  {fb:>20.2f}  {fc:>20.2f}  {fc - fb:>+7.2f}")
print("  -> identical across solvers: ordering_fixes measures the raw order, not")
print("     the route.  'extra' = avg cold items per run the raw order would have")
print("     picked before an ambient/normal one (pure cold-chain violations).")

# ===========================================================================
# 4b — Cost of the constraint: grouped vs interspersed cold zones
# ===========================================================================
print("\n=== 4b — Cold-last distance penalty: grouped vs interspersed zones ===")
print(f"  {'Layout':<22}  {'base dist':>10}  {'cold-last':>10}  {'penalty':>8}")
for label, random_zones in [("grouped (default)", False), ("interspersed", True)]:
    d = generate(n_aisles=10, n_positions=20, n_items=300, n_pickruns=500,
                 n_days=1, seed=42, random_zones=random_zones)
    g = WarehouseGraph(d.locations)
    base = _avg_dist(route_all_pickruns(d.transactions, g, d.items, solver="nn"))
    cold = _avg_dist(route_all_pickruns(d.transactions, g, d.items,
                                        locations_df=d.locations, solver="nn", cold_last=True))
    print(f"  {label:<22}  {base:>8.1f} m  {cold:>8.1f} m  {100*(cold-base)/base:>+7.1f}%")
print("  -> grouped zones: cold-last is ~free or cheaper (cold area is one block);")
print("     interspersed zones: cold-last forces zig-zagging -> real penalty.")

# ===========================================================================
# 4c — Dwell-time model in the DES
# ===========================================================================
print("\n=== 4c — Cold-item dwell time (3 pickers) ===")
tile_zone = _tile_zone(graph, ds.locations)
routes_plain = route_all_pickruns(ds.transactions, graph, ds.items, solver="nn")
routes_cold  = route_all_pickruns(ds.transactions, graph, ds.items,
                                  locations_df=ds.locations, solver="nn", cold_last=True)
if SYNC_RELEASES:
    routes_plain = [r._replace(release_s=0.0) for r in routes_plain]
    routes_cold  = [r._replace(release_s=0.0) for r in routes_cold]

print(f"  {'Routes':<22}  {'avg dwell':>10}  {'max dwell':>10}  {'cold picks':>11}")
for label, rts in [("plain NN", routes_plain), ("cold-last NN", routes_cold)]:
    s = run_des(rts, graph, n_pickers=3, zone_lookup=tile_zone)
    print(f"  {label:<22}  {s['avg_cold_dwell_s']:>8.1f} s  {s['max_cold_dwell_s']:>8.1f} s  "
          f"{s['n_cold_picks']:>11}")
print("  -> dwell = seconds a CHILLER/FREEZER item rides in the cart from pick to")
print("     pickrun end.  cold-last picks cold items last, cutting their dwell.")

# ===========================================================================
# 4d — Zone-crossing penalty: does it change which solver wins?
# ===========================================================================
print("\n=== 4d — Zone-crossing penalty (3 pickers, 60 s per crossing) ===")
PENALTY = 60.0
configs = [
    ("nn",            dict(solver="nn")),
    ("2opt",          dict(solver="2opt")),
    ("aisle_nn",      dict(solver="aisle_nn")),
    ("nn cold-last",  dict(solver="nn",  cold_last=True)),
]
print(f"  {'Routing':<16}  {'makespan(no pen)':>16}  {'makespan(+pen)':>15}  {'crossings':>10}")
for label, kw in configs:
    rts = route_all_pickruns(ds.transactions, graph, ds.items,
                             locations_df=ds.locations, solver=kw["solver"],
                             cold_last=kw.get("cold_last", False))
    if SYNC_RELEASES:
        rts = [r._replace(release_s=0.0) for r in rts]
    s0 = run_des(rts, graph, n_pickers=3)
    s1 = run_des(rts, graph, n_pickers=3,
                 zone_lookup=tile_zone, zone_cross_penalty_s=PENALTY)
    print(f"  {label:<16}  {s0['makespan_s']/60:>13.1f} min  {s1['makespan_s']/60:>12.1f} min  "
          f"{s1['n_zone_crossings']:>10}")
print("  -> without the penalty the local-search solver (2opt) tends to win on")
print("     distance; with a steep per-crossing penalty the zone-clustered")
print("     cold-last route crosses fewest boundaries and can overtake it.")

print("\nDone.")

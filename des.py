"""Discrete-event simulation of warehouse picking operations.

Model
-----
  • A pool of ``n_pickers`` pickers, each taking one pickrun at a time.
  • One-way aisles are modelled as SimPy Resources with capacity=1:
    only one picker may traverse a one-way aisle at a time.  Others
    queue at the aisle entrance until it is free.
  • Pickruns are released into the system according to their timestamps
    (``release_s`` in RouteResult).  All pickruns at t=0 if no timestamps.

Extensions (all optional, disabled by default)
----------------------------------------------
  n_replenishers      — worker pool that restocks empty slots; pickers wait
                        until a replenisher services the slot before picking.
                        Tune ``replenish_prob`` (fraction of picks that hit an
                        empty slot) and ``restock_time_s`` (seconds to restock).
  fatigue_pct_per_100 — picker walking speed degrades linearly; after every 100
                        picks the picker walks this many % slower.  Models shift-
                        end fatigue or equipment battery drain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import simpy

from graph import WarehouseGraph, DEPOT
from tsp import RouteResult, SPEED_M_S, PICK_TIME_S, _apply_solver


# ---------------------------------------------------------------------------
# Machine profiles
# ---------------------------------------------------------------------------

@dataclass
class MachineProfile:
    """Physical characteristics of a picking machine.

    Attributes
    ----------
    name
        Display name for reporting.
    speed_m_s
        Travel speed in metres/second.
    pick_time_s
        Seconds to pick one item from its slot.
    narrow_ok
        True if the machine can enter one-way (narrow) aisles and join the
        aisle queue.  False for wide machines (e.g. counterbalance forklifts)
        that must detour via the cross-aisles; they pay ``detour_factor``
        times the normal aisle travel time instead.
    detour_factor
        Only used when ``narrow_ok=False``.  Multiplier applied to the
        distance of any narrow-aisle segment to model the longer route around.
    """
    name:          str
    speed_m_s:     float
    pick_time_s:   float
    narrow_ok:     bool  = True
    detour_factor: float = 2.0


# Pre-built profiles — use directly or construct your own.
HUMAN          = MachineProfile("human",          speed_m_s=1.5,  pick_time_s=4.0)
REACH_TRUCK    = MachineProfile("reach_truck",    speed_m_s=2.5,  pick_time_s=7.0)
COUNTERBALANCE = MachineProfile("counterbalance", speed_m_s=3.5,  pick_time_s=10.0,
                                narrow_ok=False, detour_factor=2.2)
PALLET_JACK    = MachineProfile("pallet_jack",    speed_m_s=0.8,  pick_time_s=3.0)


def run_des(
    route_results: list[RouteResult],
    graph: WarehouseGraph,
    n_pickers: int = 1,
    model_aisle_contention: bool = True,
    n_replenishers: int = 0,
    replenish_prob: float = 0.0,
    restock_time_s: float = 30.0,
    fatigue_pct_per_100_picks: float = 0.0,
    machine_profile: MachineProfile | None = None,
    machine_profiles: list[MachineProfile] | None = None,
    pickrun_eligibility: list[set[str]] | None = None,
    reroute_on_wait_s: float = 0.0,
    reroute_solver: str = "nn",
    coordinated_dispatch: bool = False,
    zone_lookup: dict[str, str] | None = None,
    zone_cross_penalty_s: float = 0.0,
) -> dict:
    """Simulate all pickruns and return aggregate statistics.

    Parameters
    ----------
    route_results
        List of routed pickruns from ``tsp.route_all_pickruns()``.
    graph
        WarehouseGraph — needed for tile classification and distances.
    n_pickers
        Number of concurrent pickers sharing the warehouse.
    model_aisle_contention
        When True, odd-x (one-way) aisles have capacity=1: only one picker
        at a time, others queue.  Set False to measure the ideal lower bound
        with no aisle contention.
    n_replenishers
        Number of dedicated replenishment workers.  When > 0 and
        ``replenish_prob`` > 0, pickers occasionally find an empty slot and
        must wait for a replenisher to restock it.
    replenish_prob
        Probability (0–1) that any given pick location has an empty slot
        requiring a restock before the item can be picked.
    restock_time_s
        Time in seconds for a replenisher to service one empty slot.
    fatigue_pct_per_100_picks
        Each picker's walking speed decreases by this many percent for every
        100 picks completed.  E.g. 5.0 means after 200 picks the picker walks
        10% slower.  Speed is floored at 10% of the baseline.

    machine_profile : MachineProfile | None
        Overrides the per-picker speed and pick time.  Use the preset constants
        ``HUMAN``, ``REACH_TRUCK``, or ``COUNTERBALANCE``, or build your own
        with ``MachineProfile(...)``.  ``None`` falls back to the module-level
        ``SPEED_M_S`` / ``PICK_TIME_S`` constants (equivalent to ``HUMAN``).
        Ignored when ``machine_profiles`` is given.
    machine_profiles : list[MachineProfile] | None
        Mixed fleet (Task 3b): one profile per picker *slot*.  When given,
        ``len(machine_profiles)`` heterogeneous workers are spawned (overriding
        ``n_pickers``), each carrying its own speed / pick-time / narrow-aisle
        profile, and pickruns are dispatched from a shared pool rather than via
        anonymous resource slots.  A counterbalance worker, for instance, pays
        the detour penalty on narrow aisles while a reach-truck worker queues
        for them — within the same simulation.
    pickrun_eligibility : list[set[str]] | None
        Capability matching for the mixed fleet, parallel to ``route_results``:
        ``pickrun_eligibility[i]`` is the set of profile *names* allowed to
        serve pickrun ``i`` (e.g. ``{"counterbalance"}`` for a ground-level
        heavy run, ``{"reach_truck", "human"}`` for a high-reach run).  A worker
        only pulls a pickrun whose eligibility set contains its profile name.
        ``None`` means any worker may serve any pickrun.  Only used when
        ``machine_profiles`` is given.
    reroute_on_wait_s : float
        When > 0 and a picker arrives at a one-way aisle that already has
        waiters, they re-optimise their remaining non-blocked tiles with
        ``reroute_solver`` and push the blocked aisle to the end of their
        route.  Set to e.g. 30.0 to explore whether opportunistic rerouting
        reduces makespan.
    reroute_solver : str
        TSP solver used to re-order the un-blocked tiles when a reroute
        fires (``reroute_on_wait_s > 0``).  Any solver accepted by
        ``tsp._apply_solver`` is valid; defaults to ``"nn"``.  Ignored
        when ``reroute_on_wait_s == 0``.
    coordinated_dispatch : bool
        When True a shared work-pool replaces per-pickrun pre-assigned
        routes.  A central dispatcher process monitors aisle queues every
        10 s and pushes tiles from congested aisles to the back of the
        pool.  Exactly ``n_pickers`` workers draw from the pool until it
        is empty.  Incompatible with per-pickrun release times (all tiles
        start at t=0).
    zone_lookup : dict[str, str] | None
        ``{tile_id: zone_type_1}`` map.  When provided the DES becomes
        temperature-aware:
          • Cold-chain dwell time (4c) is tracked for every CHILLER/FREEZER
            pick — the seconds from when the cold item is picked until its
            pickrun finishes (it stays in the cart that whole time).  Surfaced
            in the ``*_cold_dwell_s`` return keys.
          • Enables the zone-crossing penalty when ``zone_cross_penalty_s > 0``.
        Not tracked in ``coordinated_dispatch`` mode (no per-pickrun grouping).
    zone_cross_penalty_s : float
        Seconds added (``env.timeout``) whenever two consecutive picked tiles
        belong to different zones (4d) — models the handling/door overhead of
        moving between temperature areas.  Requires ``zone_lookup``.  Routes
        that cluster picks by zone (e.g. ``cold_last=True``) cross fewer zone
        boundaries and so pay less penalty.

    Returns
    -------
    dict with keys:
        makespan_s              — wall-clock time until the last pickrun finishes
        avg_pickrun_time_s      — mean time a picker spends on one pickrun
        avg_one_way_wait_s      — mean wait time per aisle-entry event
        worst_aisle_x           — x-column with the longest total wait (or None)
        worst_aisle_wait_s      — total wait accumulated in worst_aisle_x
        total_restock_wait_s    — total seconds all pickers spent waiting for restocks
        avg_restock_wait_s      — mean restock-wait per individual restock event
        n_restock_events        — number of restock waits that occurred
        n_reroutes              — number of times a picker was rerouted mid-run
        avg_cold_dwell_s        — mean cold-item dwell time (0 without zone_lookup)
        max_cold_dwell_s        — longest single cold-item dwell time
        total_cold_dwell_s      — sum of all cold-item dwell times
        n_cold_picks            — number of CHILLER/FREEZER picks observed
        n_zone_crossings        — number of consecutive-tile zone changes penalised
    """
    mp_default = machine_profile or HUMAN
    env       = simpy.Environment()
    oneway_xs = graph.oneway_xs()
    rng       = np.random.default_rng(0)

    # Temperature awareness (Task 4c/4d).  A tile is "cold" if its zone is
    # CHILLER or FREEZER.  zone_of() returns None for the depot / unmapped tiles.
    zone_lookup = zone_lookup or {}

    def zone_of(tile: str) -> str | None:
        return zone_lookup.get(tile)

    cold_dwells:    list[float] = []   # 4c: per cold-pick dwell times (seconds)
    zone_crossings: list[int]   = [0]  # 4d: penalised consecutive-tile crossings

    picker_pool = simpy.Resource(env, capacity=max(1, n_pickers))
    aisle_res: dict[int, simpy.Resource] = (
        {x: simpy.Resource(env, capacity=1) for x in oneway_xs}
        if model_aisle_contention
        else {}
    )

    # Replenishment: a Store holds pending restock events; replenisher workers
    # pull one event, sleep for restock_time_s, then trigger the event so the
    # waiting picker can proceed.
    restock_queue: simpy.Store = simpy.Store(env)

    aisle_wait:    dict[int, list[float]] = {x: [] for x in oneway_xs}
    pickrun_times: list[float] = []
    restock_waits: list[float] = []
    reroute_count:   list[int]   = [0]   # mutable counter shared across closures
    dispatch_finish: list[float] = []   # absolute finish times for coordinated pickers

    # --- Replenisher worker processes ---
    def _replenisher_worker():
        while True:
            done_event = yield restock_queue.get()
            yield env.timeout(restock_time_s)
            done_event.succeed()

    if n_replenishers > 0 and replenish_prob > 0.0:
        for _ in range(n_replenishers):
            env.process(_replenisher_worker())

    # --- Picker processes ---
    def picker_process(rr: RouteResult):
        yield env.timeout(rr.release_s)
        with picker_pool.request() as req:
            yield req
            yield from _do_route(rr, mp_default)

    def _do_route(rr: RouteResult, mp: MachineProfile):
        t_start    = env.now
        tiles      = list(rr.route_tiles)   # mutable copy so rerouting can reorder
        prev       = tiles[0]
        picks_done = 0
        i          = 1
        cold_pick_times: list[float] = []   # 4c: when each cold item entered cart

        def _arrive(from_tile: str, to_tile: str):
            """Cross-zone penalty (4d) + pick + cold-dwell bookkeeping (4c)."""
            if zone_cross_penalty_s > 0.0:
                zf, zt = zone_of(from_tile), zone_of(to_tile)
                if zf is not None and zt is not None and zf != zt:
                    zone_crossings[0] += 1
                    yield env.timeout(zone_cross_penalty_s)
            yield from _pick_at(to_tile, mp.pick_time_s)
            if zone_of(to_tile) in ("CHILLER", "FREEZER"):
                cold_pick_times.append(env.now)

        while i < len(tiles):
            tile = tiles[i]
            x    = graph.tile_x(tile)

            # Speed: machine profile base, then linearly degraded by fatigue
            eff_speed = mp.speed_m_s
            if fatigue_pct_per_100_picks > 0.0:
                reduction = (picks_done / 100.0) * (fatigue_pct_per_100_picks / 100.0)
                eff_speed = mp.speed_m_s * max(0.1, 1.0 - reduction)

            if x is not None and x in aisle_res:
                # Collect the full contiguous segment in this one-way aisle
                seg: list[str] = []
                j = i
                while j < len(tiles) and graph.tile_x(tiles[j]) == x:
                    seg.append(tiles[j])
                    j += 1

                if not mp.narrow_ok:
                    # Wide machine: can't enter — pay a detour penalty instead
                    # of queuing for the resource.
                    for t in seg:
                        d = graph.distance(prev, t) * mp.detour_factor
                        yield env.timeout(d / eff_speed)
                        if t != graph.end_tile:
                            yield from _arrive(prev, t)
                            picks_done += 1
                        prev = t
                    i = j
                else:
                    # Narrow-capable machine: queue for the one-way aisle.
                    # When reroute_on_wait_s > 0 the picker reneges if it has
                    # waited longer than the threshold AND has other (non-blocked)
                    # tiles it could pick meanwhile.  On renege it re-solves the
                    # un-blocked tiles with reroute_solver and defers this aisle to
                    # the end of the route.  This models genuine wait-then-give-up,
                    # so the threshold value actually governs behaviour.
                    rest_picks = [t for t in tiles[i:]
                                  if graph.tile_x(t) != x and t != graph.end_tile]
                    use_reroute = reroute_on_wait_s > 0 and len(rest_picks) > 0
                    t_req = env.now
                    with aisle_res[x].request() as req:
                        if use_reroute:
                            got = req in (yield req | env.timeout(reroute_on_wait_s))
                        else:
                            yield req
                            got = True

                        if got:
                            aisle_wait[x].append(env.now - t_req)
                            for t in seg:
                                yield env.timeout(graph.distance(prev, t) / eff_speed)
                                if t != graph.end_tile:
                                    yield from _arrive(prev, t)
                                    picks_done += 1
                                prev = t
                            i = j
                        else:
                            # Waited reroute_on_wait_s without acquiring the aisle:
                            # give up, re-solve the un-blocked tiles, defer this
                            # aisle.  i is left unchanged so the reordered tiles[i]
                            # is re-evaluated on the next loop iteration.
                            deferred = [t for t in tiles[i:] if graph.tile_x(t) == x]
                            end_part = ([graph.end_tile]
                                        if graph.end_tile in tiles[i:] else [])
                            solved, _ = _apply_solver(reroute_solver, rest_picks,
                                                      prev, graph)
                            tiles[i:] = solved + deferred + end_part
                            reroute_count[0] += 1
            else:
                yield env.timeout(graph.distance(prev, tile) / eff_speed)
                if tile != graph.end_tile:
                    yield from _arrive(prev, tile)
                    picks_done += 1
                prev = tile
                i   += 1

        pickrun_times.append(env.now - t_start)
        # 4c: each cold item dwelled from its pick until the pickrun finished.
        t_end = env.now
        for tp in cold_pick_times:
            cold_dwells.append(t_end - tp)

    def _pick_at(tile: str, pick_time_s: float):
        """Yield pick-time events, inserting a restock wait when the slot is empty.

        ``pick_time_s`` comes from the active picker's MachineProfile, so a
        reach truck (7 s) and a human (4 s) spend different times at each slot.
        """
        if (n_replenishers > 0
                and replenish_prob > 0.0
                and rng.random() < replenish_prob):
            t_req      = env.now
            done_event = env.event()
            yield restock_queue.put(done_event)
            yield done_event
            restock_waits.append(env.now - t_req)
        yield env.timeout(pick_time_s)

    # --- Coordinated dispatch (2c) — shared work-pool + central dispatcher ---
    if coordinated_dispatch:
        work_pool: list[str] = [
            t for rr in route_results
            for t in rr.route_tiles
            if t not in (graph.start_tile, graph.end_tile)
        ]

        def _coordinated_picker(_picker_id: int):
            cur        = graph.start_tile
            t_start    = env.now
            picks_done = 0
            while work_pool:
                idx  = min(range(len(work_pool)),
                           key=lambda i: graph.distance(cur, work_pool[i]))
                tile = work_pool.pop(idx)
                x    = graph.tile_x(tile)

                eff_speed = mp_default.speed_m_s
                if fatigue_pct_per_100_picks > 0.0:
                    reduction = (picks_done / 100.0) * (fatigue_pct_per_100_picks / 100.0)
                    eff_speed = mp_default.speed_m_s * max(0.1, 1.0 - reduction)

                if x is not None and x in aisle_res:
                    # Grab every available tile in this aisle at once so we
                    # only request the resource once per aisle visit.
                    same = [t for t in work_pool if graph.tile_x(t) == x]
                    for t in same:
                        work_pool.remove(t)
                    seg = sorted([tile] + same,
                                 key=lambda t: int(t.split("_")[1]))

                    if not mp_default.narrow_ok:
                        for t in seg:
                            d = graph.distance(cur, t) * mp_default.detour_factor
                            yield env.timeout(d / eff_speed)
                            yield from _pick_at(t, mp_default.pick_time_s)
                            picks_done += 1
                            cur = t
                    else:
                        t_req = env.now
                        with aisle_res[x].request() as req:
                            yield req
                            aisle_wait[x].append(env.now - t_req)
                            for t in seg:
                                yield env.timeout(graph.distance(cur, t) / eff_speed)
                                yield from _pick_at(t, mp_default.pick_time_s)
                                picks_done += 1
                                cur = t
                else:
                    yield env.timeout(graph.distance(cur, tile) / eff_speed)
                    yield from _pick_at(tile, mp_default.pick_time_s)
                    picks_done += 1
                    cur = tile

            yield env.timeout(graph.distance(cur, graph.end_tile) / eff_speed)
            dispatch_finish.append(env.now)
            pickrun_times.append(env.now - t_start)

        def _dispatcher():
            """Periodically deprioritise tiles in congested aisles."""
            while work_pool:
                yield env.timeout(10.0)
                for x, res in aisle_res.items():
                    if len(res.queue) > 0:
                        congested = [t for t in work_pool if graph.tile_x(t) == x]
                        if congested:
                            rest = [t for t in work_pool if graph.tile_x(t) != x]
                            work_pool[:] = rest + congested
                            reroute_count[0] += len(congested)

        for pid in range(n_pickers):
            env.process(_coordinated_picker(pid))
        if model_aisle_contention:
            env.process(_dispatcher())
    elif machine_profiles is not None:
        # --- Mixed fleet (3b) — N heterogeneous workers from a shared pool ---
        # Each worker carries its own MachineProfile.  Pickruns are fed into a
        # FilterStore at their release time; a worker only pulls a pickrun whose
        # eligibility set contains the worker's profile name.
        fleet = machine_profiles
        elig  = pickrun_eligibility
        store: simpy.FilterStore = simpy.FilterStore(env)
        remaining = [len(route_results)]

        # Fail loudly if a pickrun can never be served by the fleet on offer
        # (otherwise it would sit in the FilterStore forever and be silently
        # dropped from the makespan).
        if elig is not None:
            fleet_names = {p.name for p in fleet}
            orphan = [i for i, e in enumerate(elig) if not (e & fleet_names)]
            if orphan:
                raise ValueError(
                    f"{len(orphan)} pickrun(s) have eligibility sets no profile in "
                    f"the fleet {sorted(fleet_names)} can satisfy "
                    f"(e.g. pickrun {orphan[0]} needs {sorted(elig[orphan[0]])})."
                )

        def _feeder():
            order = sorted(range(len(route_results)),
                           key=lambda i: route_results[i].release_s)
            clock = 0.0
            for i in order:
                rr = route_results[i]
                if rr.release_s > clock:
                    yield env.timeout(rr.release_s - clock)
                    clock = rr.release_s
                yield store.put((i, rr))

        def _fleet_worker(wmp: MachineProfile):
            while remaining[0] > 0:
                if elig is None:
                    item = yield store.get()
                else:
                    item = yield store.get(lambda it: wmp.name in elig[it[0]])
                remaining[0] -= 1
                yield from _do_route(item[1], wmp)

        env.process(_feeder())
        for wmp in fleet:
            env.process(_fleet_worker(wmp))
    else:
        for rr in route_results:
            env.process(picker_process(rr))

    env.run()

    makespan  = float(max(dispatch_finish) if coordinated_dispatch and dispatch_finish
                      else env.now)
    avg_time  = float(np.mean(pickrun_times)) if pickrun_times else 0.0
    all_waits = [w for ws in aisle_wait.values() for w in ws]
    avg_wait  = float(np.mean(all_waits)) if all_waits else 0.0

    worst_x:    int | None = None
    worst_wait: float      = 0.0
    if aisle_wait:
        totals = {x: sum(ws) for x, ws in aisle_wait.items() if ws}
        if totals:
            worst_x    = max(totals, key=totals.get)
            worst_wait = totals[worst_x]

    total_rw = sum(restock_waits)
    avg_rw   = float(np.mean(restock_waits)) if restock_waits else 0.0

    return {
        "makespan_s":           makespan,
        "avg_pickrun_time_s":   avg_time,
        "avg_one_way_wait_s":   avg_wait,
        "worst_aisle_x":        worst_x,
        "worst_aisle_wait_s":   worst_wait,
        "total_restock_wait_s": total_rw,
        "avg_restock_wait_s":   avg_rw,
        "n_restock_events":     len(restock_waits),
        "n_reroutes":           reroute_count[0],
        "avg_cold_dwell_s":     float(np.mean(cold_dwells)) if cold_dwells else 0.0,
        "max_cold_dwell_s":     float(max(cold_dwells)) if cold_dwells else 0.0,
        "total_cold_dwell_s":   float(sum(cold_dwells)),
        "n_cold_picks":         len(cold_dwells),
        "n_zone_crossings":     zone_crossings[0],
    }

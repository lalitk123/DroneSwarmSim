"""Map generalization of the relay-anchoring head-to-head (Experiment 2).

Question: is worst-gap anchoring's win (relay_baseline_comparison.py) a property of the
*rule* or of the one 40x40 map used so far? Rerun the 5-method head-to-head
(no_relay, worst_gap, centroid, centfar, fiedler; K=4 relays, multihop routing ON,
centralized base strategy, 10 paired seeds) across a grid of map conditions:

(a) Map sizes 40x40 / 60x60 / 100x100. Drone count scales roughly with linear size
    (12 / 18 / 30 — constant fleet-per-diagonal rather than per-area, which at 75
    drones would trivialize the comm problem), victims = drones, obstacles scale with
    area (25 / 56 / 156), max_steps with area-ish (600 / 900 / 1500), and the disk
    comm radii scale proportionally to the map diagonal (x1.0 / x1.5 / x2.5 of the
    original [18, 12, 9, 6, 4]) so severity is comparable across sizes.

(b) Dead-zone variants on the 40x40 base config, each combined with the same disk
    radius sweep (Composite[RangeComm, DeadZone(block_crossing=True)]):
    (i)   dz_default — the project's existing default dead-zone layout, i.e. the
          mid-height wall from relay_multibranch_experiment.py (_zone_rects, 60% of
          the width blocked, gap on the right), scaled from 50x50 to 40x40:
          rect (17, 0, 23, 24).
    (ii)  dz_rand1/2/3 — three randomized dead-zone placements (fixed placement seeds
          1, 2, 3; two random ~7x12 rectangles each, never covering the base), held
          constant across episode seeds within a condition.
    (iii) NO dead zones at all — this is exactly the original relay_baseline_comparison
          condition (labelled 40x40 here) and doubles as the 40x40 size point. It is
          the critical control: does worst_gap still beat centroid/centfar when the
          "centroid falls in the dead zone" trap does not exist?

Paired seeding: within a condition every method sees the identical cfg.seed (300+s)
and comm rng (seed+9999) per seed s, mirroring relay_baseline_comparison.py exactly.
Per the existing harness, cfg.comm_radius stays at the condition's *nominal* (most
generous) radius while the tested RangeComm radius varies — this preserves the
published semantics of CENTFAR's "far" threshold and Fiedler's sigma.

Run (chunkable; appends to the CSV and skips already-done rows):
    python -m swarm_sim.relay_map_generalization --conditions 40x40,60x60
    python -m swarm_sim.relay_map_generalization            # all conditions
    python -m swarm_sim.relay_map_generalization --report   # summary + plot only
Outputs: results/relay_map_generalization.csv, results/relay_map_generalization.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .relay import RelayStrategy
from .relay_baselines import (CentroidRelayStrategy, CentroidOfFarRelayStrategy,
                              FiedlerRelayStrategy)
from .comm import RangeComm, DeadZone, Composite
from .metrics import resilience_auc

N_SEEDS = 10
N_RELAYS = 4
SEED0 = 300           # same seed base as relay_baseline_comparison.py
BASE_RADII = [18, 12, 9, 6, 4]

CSV_PATH = os.path.join("results", "relay_map_generalization.csv")
PNG_PATH = os.path.join("results", "relay_map_generalization.png")
FIELDS = ["condition", "variant", "disk_radius", "seed",
          "coverage", "detection", "connectivity", "mission_time"]

VARIANTS = {
    "no_relay":  lambda: CentralizedStrategy(),
    "centroid":  lambda: CentroidRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "centfar":   lambda: CentroidOfFarRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "fiedler":   lambda: FiedlerRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "worst_gap": lambda: RelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
}


def _size_cfg(size, scale, n_drones, n_obstacles, clusters, max_steps):
    radii = [r * scale for r in BASE_RADII]
    return dict(cfg=dict(width=size, height=size, n_obstacles=n_obstacles,
                         obstacle_clusters=clusters, n_drones=n_drones,
                         n_victims=n_drones, max_steps=max_steps, seed=SEED0,
                         comm_radius=radii[0]),
                radii=radii, zones=None)


def _random_zones(placement_seed, width=40, height=40, n_zones=2, base=(0, 0)):
    """Two random ~7x12-ish rectangular dead zones; never covering the base cell.
    Layout is a function of placement_seed only, constant across episode seeds."""
    rng = np.random.default_rng(1000 + placement_seed)
    zones = []
    while len(zones) < n_zones:
        h = int(rng.integers(5, 9))       # 5..8 rows tall
        w = int(rng.integers(9, 15))      # 9..14 cols wide
        y0 = int(rng.integers(0, height - h))
        x0 = int(rng.integers(0, width - w))
        rect = (y0, x0, y0 + h, x0 + w)
        by, bx = base
        if rect[0] <= by <= rect[2] and rect[1] <= bx <= rect[3]:
            continue  # never swallow the base station
        zones.append(rect)
    return zones


_BASE40 = dict(width=40, height=40, n_obstacles=25, obstacle_clusters=6,
               n_drones=12, n_victims=12, max_steps=600, seed=SEED0,
               comm_radius=18.0)   # identical to relay_baseline_comparison BASE

CONDITIONS = {
    # (a) map sizes — no dead zones; 40x40 == the original head-to-head condition
    "40x40":   _size_cfg(40, 1.0, 12, 25, 6, 600),
    "60x60":   _size_cfg(60, 1.5, 18, 56, 9, 900),
    "100x100": _size_cfg(100, 2.5, 30, 156, 16, 1500),
    # (b) dead-zone variants on the 40x40 base config
    # default layout = relay_multibranch_experiment._zone_rects wall (60% of width,
    # +-3 rows around mid-height, gap on the right), scaled 50->40.
    "dz_default": dict(cfg=dict(_BASE40), radii=list(map(float, BASE_RADII)),
                       zones=[(17, 0, 23, 24)]),
    "dz_rand1": dict(cfg=dict(_BASE40), radii=list(map(float, BASE_RADII)),
                     zones=_random_zones(1)),
    "dz_rand2": dict(cfg=dict(_BASE40), radii=list(map(float, BASE_RADII)),
                     zones=_random_zones(2)),
    "dz_rand3": dict(cfg=dict(_BASE40), radii=list(map(float, BASE_RADII)),
                     zones=_random_zones(3)),
}


def _run_one(cond_name, variant, radius, seed_offset):
    spec = CONDITIONS[cond_name]
    cfg = SimConfig(**{**spec["cfg"], "seed": SEED0 + seed_offset})
    rng = np.random.default_rng(cfg.seed + 9999)   # identical rng across methods
    comm = RangeComm(radius=radius, rng=rng, mode="disk")
    if spec["zones"]:
        comm = Composite([comm, DeadZone(spec["zones"], block_crossing=True)])
    m = Simulator(cfg, VARIANTS[variant](), comm=comm, multihop=True).run()
    return m


def _done_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add((row["condition"], row["variant"],
                          float(row["disk_radius"]), int(row["seed"])))
    return done


def run_conditions(cond_names):
    os.makedirs("results", exist_ok=True)
    done = _done_keys(CSV_PATH)
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            f.flush()
        for cond in cond_names:
            spec = CONDITIONS[cond]
            for radius in spec["radii"]:
                for variant in VARIANTS:
                    t0 = time.time()
                    n_new = 0
                    for s in range(N_SEEDS):
                        key = (cond, variant, float(radius), SEED0 + s)
                        if key in done:
                            continue
                        m = _run_one(cond, variant, radius, s)
                        w.writerow(dict(condition=cond, variant=variant,
                                        disk_radius=radius, seed=SEED0 + s,
                                        coverage=round(m.coverage, 4),
                                        detection=round(m.detection_rate, 4),
                                        connectivity=round(m.mean_connectivity, 4),
                                        mission_time=m.mission_time))
                        f.flush()
                        n_new += 1
                    if n_new:
                        print(f"[{cond}] r={radius:g} {variant}: {n_new} runs "
                              f"in {time.time()-t0:.1f}s", flush=True)


# ---- reporting ---------------------------------------------------------------

def _load():
    data = {}   # cond -> variant -> radius -> list of coverage / conn
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            c, v, r = row["condition"], row["variant"], float(row["disk_radius"])
            d = data.setdefault(c, {}).setdefault(v, {}).setdefault(
                r, {"cov": [], "conn": []})
            d["cov"].append(float(row["coverage"]))
            d["conn"].append(float(row["connectivity"]))
    return data


def report():
    data = _load()
    summary = {}   # cond -> variant -> (radii, mean covs, auc)
    for cond, spec in CONDITIONS.items():
        if cond not in data:
            continue
        radii = spec["radii"]
        sev = [(radii[0] - r) / (radii[0] - radii[-1]) for r in radii]
        summary[cond] = {}
        print(f"\n=== {cond} (radii {['%g' % r for r in radii]}) ===")
        print("variant".ljust(11) + "".join(f"{('r=%g' % r):>9}" for r in radii)
              + f"{'AUC':>9}")
        for v in VARIANTS:
            covs = [float(np.mean(data[cond][v][r]["cov"])) for r in radii]
            auc = resilience_auc(sev, covs, baseline=covs[0])
            summary[cond][v] = (radii, covs, auc)
            print(v.ljust(11) + "".join(f"{c:>9.3f}" for c in covs) + f"{auc:>9.3f}")
        ranking = sorted(summary[cond], key=lambda v: -summary[cond][v][2])
        print("AUC ranking: " + " > ".join(
            f"{v}({summary[cond][v][2]:.3f})" for v in ranking))
    _plot(data, summary)
    return summary


def _plot(data, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    # validated categorical palette (fixed slot order), one style per method
    colors = {"no_relay": "#898781", "worst_gap": "#2a78d6", "centroid": "#eda100",
              "centfar": "#008300", "fiedler": "#4a3aa7"}
    markers = {"no_relay": "x", "worst_gap": "o", "centroid": "^",
               "centfar": "d", "fiedler": "s"}
    lstyle = {"no_relay": ":", "worst_gap": "-", "centroid": "-",
              "centfar": "-", "fiedler": "-"}
    conds = [c for c in CONDITIONS if c in data]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.2))
    axes = axes.ravel()
    for ax in axes[len(conds):]:
        ax.set_visible(False)
    for ax, cond in zip(axes, conds):
        radii = CONDITIONS[cond]["radii"]
        sev = [(radii[0] - r) / (radii[0] - radii[-1]) for r in radii]
        for v in VARIANTS:
            covs = [float(np.mean(data[cond][v][r]["cov"])) for r in radii]
            ax.plot(sev, covs, lstyle[v], color=colors[v], marker=markers[v],
                    markersize=5, linewidth=1.8,
                    label=f"{v} (AUC {summary[cond][v][2]:.2f})")
        ax.set_title(cond, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("range severity (0 = generous, 1 = worst radius)",
                      fontsize=8, color="#52514e")
        ax.set_ylabel("coverage", fontsize=9, color="#52514e")
        ax.set_ylim(0, 1.05)
        ax.grid(True, color="#e1e0d9", linewidth=0.7)
        ax.tick_params(labelsize=8, colors="#898781")
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    fig.suptitle("Relay-anchoring map generalization: coverage vs comm-range severity "
                 f"(K={N_RELAYS} relays, multihop, centralized base, {N_SEEDS} paired seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PNG_PATH, dpi=120, facecolor="white")
    print(f"\nSaved plot: {PNG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset of: " + ",".join(CONDITIONS))
    ap.add_argument("--report", action="store_true",
                    help="only summarize/plot existing CSV")
    args = ap.parse_args()
    if not args.report:
        names = list(CONDITIONS) if not args.conditions \
            else [c.strip() for c in args.conditions.split(",")]
        for c in names:
            if c not in CONDITIONS:
                sys.exit(f"unknown condition: {c}")
        run_conditions(names)
    report()


if __name__ == "__main__":
    main()

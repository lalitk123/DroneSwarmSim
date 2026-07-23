"""EXPERIMENT 4 — Fleet-size scaling and the single-anchor limit.

Tests whether single-anchor worst-gap relay anchoring (RelayStrategy) still works as
the searcher swarm grows and spreads out, versus MultiBranchRelayStrategy (k-means
clusters searchers into branches, one relay chain per branch) and the CENTFAR-style
literature baseline, on a 60x60 map (2.25x the area of the 40x40 map used in
relay_baseline_comparison.py).

Design (mirrors relay_baseline_comparison.py so results are directly comparable):
- Map 60x60; obstacle/victim density scaled by area from the 40x40 config
  (25 obst / 6 clusters / 12 victims  ->  56 obst / 13 clusters / 27 victims).
- Searcher fleet size in {12, 24, 36}; K in {4, 6} relays; total n_drones = fleet + K
  for every variant including no_relay (so no_relay gets the K extra drones as
  searchers — same total fleet, apples-to-apples).
- Disk comm radii [18, 13, 9]: the 40x40 sweep used [18,12,9,6,4]; map linear scale
  is 60/40 = 1.5, so radii 12/9/6 (where relays mattered most on 40x40: big gains at
  12 and 9, tapering by 6) map onto 18/13.5/9 -> rounded to [18, 13, 9].
- 10 seeds, paired: identical cfg seed per seed index across all variants.
- multihop=True, centralized base strategy, RangeComm disk mode — as in the original.
- One deliberate difference from the original harness: cfg.comm_radius is set to the
  *swept* radius (the original left it at 18 while sweeping RangeComm down to 4,
  which silently degrades CENTFAR's "far" threshold into plain centroid at small
  radii). Setting it keeps CENTFAR fair, i.e. not a strawman.
- MultiBranch branches: n_branches = 2 for K=4 and 3 for K=6 (>= 2 relays/branch);
  no dead zones on this map, so its advantage must come purely from splitting the
  chain across spatially separated searcher clusters.

Run: python -m swarm_sim.relay_fleet_scaling            (full sweep, resumable)
     python -m swarm_sim.relay_fleet_scaling --summary  (re-summarize/plot from CSV)
Outputs: results/relay_fleet_scaling.csv (per-seed rows, appended incrementally),
         results/relay_fleet_scaling.png
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .relay import RelayStrategy, MultiBranchRelayStrategy
from .relay_baselines import CentroidOfFarRelayStrategy
from .comm import RangeComm
from .metrics import resilience_auc

BASE = dict(width=60, height=60, n_obstacles=56, obstacle_clusters=13,
            n_victims=27, max_steps=900, seed=300)
N_SEEDS = 10
RADII = [18, 13, 9]
FLEETS = [12, 24, 36]          # searcher counts (relays are on top of these)
KS = [4, 6]
METHODS = ["no_relay", "worst_gap", "multi_branch", "centfar"]

CSV_PATH = os.path.join("results", "relay_fleet_scaling.csv")
PNG_PATH = os.path.join("results", "relay_fleet_scaling.png")
FIELDS = ["method", "fleet", "k", "radius", "seed",
          "coverage", "detection", "connectivity", "mission_time"]


def n_branches_for(k):
    return 2 if k <= 4 else 3


def make_strategy(method, k):
    if method == "no_relay":
        return CentralizedStrategy()
    if method == "worst_gap":
        return RelayStrategy(CentralizedStrategy(), n_relays=k)
    if method == "multi_branch":
        return MultiBranchRelayStrategy(CentralizedStrategy(), n_relays=k,
                                        dead_zones=None,
                                        n_branches=n_branches_for(k))
    if method == "centfar":
        return CentroidOfFarRelayStrategy(CentralizedStrategy(), n_relays=k)
    raise ValueError(method)


def run_one(method, fleet, k, radius, seed_idx):
    cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + seed_idx,
                       "n_drones": fleet + k, "comm_radius": float(radius)})
    rng = np.random.default_rng(cfg.seed + 9999)
    comm = RangeComm(radius=radius, rng=rng, mode="disk")
    m = Simulator(cfg, make_strategy(method, k), comm=comm, multihop=True).run()
    return dict(method=method, fleet=fleet, k=k, radius=radius, seed=seed_idx,
                coverage=round(m.coverage, 4), detection=round(m.detection_rate, 4),
                connectivity=round(m.mean_connectivity, 4),
                mission_time=m.mission_time)


def load_done():
    done = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                key = (row["method"], int(row["fleet"]), int(row["k"]),
                       int(row["radius"]), int(row["seed"]))
                done[key] = row
    return done


def main():
    os.makedirs("results", exist_ok=True)
    done = load_done()
    new_file = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
        f.flush()

    combos = [(m, fl, k, r, s)
              for fl in FLEETS for k in KS for r in RADII
              for m in METHODS for s in range(N_SEEDS)]
    todo = [c for c in combos if c not in done]
    print(f"{len(done)} runs already in CSV; {len(todo)} to go.")
    t0 = time.time()
    for i, (m, fl, k, r, s) in enumerate(todo):
        row = run_one(m, fl, k, r, s)
        w.writerow(row)
        f.flush()
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] {m} fleet={fl} K={k} r={r} seed={s} "
                  f"({el:.0f}s elapsed, {el/(i+1):.1f}s/run)", flush=True)
    f.close()
    summarize_and_plot()


def summarize_and_plot():
    done = load_done()
    # mean coverage per (fleet, k, method, radius)
    def mean_cov(m, fl, k, r):
        vals = [float(done[(m, fl, k, r, s)]["coverage"]) for s in range(N_SEEDS)]
        return float(np.mean(vals))

    def mean_conn(m, fl, k, r):
        vals = [float(done[(m, fl, k, r, s)]["connectivity"]) for s in range(N_SEEDS)]
        return float(np.mean(vals))

    sev = [(RADII[0] - r) / (RADII[0] - RADII[-1]) for r in RADII]
    print("\n60x60 map, multihop ON, centralized base strategy, "
          f"{N_SEEDS} seeds, radii {RADII}")
    summary = {}
    for fl in FLEETS:
        for k in KS:
            print(f"\n--- fleet={fl} searchers, K={k} relays "
                  f"(total {fl+k} drones) ---")
            print("radius".ljust(8) + "".join(f"{m:>14}" for m in METHODS))
            covs = {m: [mean_cov(m, fl, k, r) for r in RADII] for m in METHODS}
            for i, r in enumerate(RADII):
                print(f"{r:<8}" + "".join(f"{covs[m][i]:>14.3f}" for m in METHODS))
            print("AUC*".ljust(8) + "".join(
                f"{resilience_auc(sev, covs[m], baseline=covs[m][0]):>14.3f}"
                for m in METHODS))
            print("absAUC".ljust(8) + "".join(
                f"{resilience_auc(sev, covs[m], baseline=1.0):>14.3f}"
                for m in METHODS))
            for m in METHODS:
                summary[(fl, k, m)] = dict(
                    cov=covs[m],
                    auc=resilience_auc(sev, covs[m], baseline=covs[m][0]),
                    abs_auc=resilience_auc(sev, covs[m], baseline=1.0))
    print("\nAUC* is SELF-BASELINED (baseline = that method's own radius-18 "
          "coverage): it measures flatness/retention, NOT performance. A method "
          "with no working radius-18 anchor (no_relay covers only ~0.09-0.24) can "
          "score a high AUC* simply by being uniformly bad, so AUC* must not be "
          "used to rank no_relay against the relay methods. Use absAUC "
          "(baseline=1.0, i.e. area under the raw coverage curve) or the "
          "coverage rows for cross-method ranking.")

    # paired per-seed multi_branch - worst_gap deltas
    print("\nPaired per-seed coverage delta (multi_branch - worst_gap), "
          "mean +/- sd over seeds:")
    for fl in FLEETS:
        for k in KS:
            for r in RADII:
                d = [float(done[("multi_branch", fl, k, r, s)]["coverage"]) -
                     float(done[("worst_gap", fl, k, r, s)]["coverage"])
                     for s in range(N_SEEDS)]
                n_pos = sum(1 for x in d if x > 0)
                print(f"  fleet={fl:<3} K={k} r={r:<3} "
                      f"delta={np.mean(d):+.3f} +/- {np.std(d):.3f} "
                      f"(mb better in {n_pos}/{N_SEEDS} seeds)")
    _plot(summary)


def _plot(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    # validated categorical palette (dataviz skill): baseline is neutral,
    # methods get fixed hues; identity also carried by marker + linestyle.
    style = {
        "no_relay":     dict(color="#6b6a63", ls=":",  marker="x"),
        "worst_gap":    dict(color="#2a78d6", ls="--", marker="o"),
        "multi_branch": dict(color="#1baf7a", ls="-",  marker="s"),
        "centfar":      dict(color="#eda100", ls="-.", marker="d"),
    }
    fig, axes = plt.subplots(len(KS), len(FLEETS),
                             figsize=(4.0 * len(FLEETS), 3.4 * len(KS)),
                             sharex=True, sharey=True)
    for ri, k in enumerate(KS):
        for ci, fl in enumerate(FLEETS):
            ax = axes[ri][ci]
            for m in METHODS:
                st = style[m]
                ax.plot(RADII, summary[(fl, k, m)]["cov"], label=m,
                        color=st["color"], linestyle=st["ls"],
                        marker=st["marker"], markersize=5, linewidth=1.8)
            ax.set_title(f"fleet={fl} searchers, K={k} relays", fontsize=10)
            ax.invert_xaxis()
            ax.set_ylim(0, 1.05)
            ax.set_xticks(RADII)
            ax.grid(True, alpha=0.25)
            if ri == len(KS) - 1:
                ax.set_xlabel("disk comm radius (smaller = worse)")
            if ci == 0:
                ax.set_ylabel("coverage at mission end")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               frameon=False)
    fig.suptitle("Fleet-size scaling: single-anchor worst-gap vs multi-branch relays "
                 "(60x60, multihop, centralized base)", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(PNG_PATH, dpi=120)
    print(f"\nSaved plot: {PNG_PATH}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarize_and_plot()
    else:
        main()

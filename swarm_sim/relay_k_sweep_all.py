"""K sweep across ALL relay-repositioning methods (Experiment 3).

The head-to-head baseline comparison (`relay_baseline_comparison.py`) fixed K=4. This
sweeps K in {2, 4, 6, 8} for all four relay methods — worst_gap, centroid, centfar,
fiedler — plus the no-relay reference, under the identical standard config: 40x40 map,
fleet of 12 drones, multihop routing ON, centralized base strategy, disk comm radius
swept over [18, 12, 9, 6, 4], 10 seeds (paired: every method/K sees the same seeds and
the same comm RNG stream).

Fleet convention — CARVE-OUT (the existing convention in relay_k_sweep.py and
relay_baseline_comparison.py): the fleet is fixed at 12 drones and K of them are
reserved as relays, so searchers = 12 - K. The no_relay reference therefore has 12
searchers while K=8 has only 4; the K sweep measures the connectivity-vs-search-
throughput tradeoff, exactly as in relay_k_sweep.py.

Key question: does Fiedler (a diffuse global algebraic-connectivity objective) close
the gap to worst-gap anchoring when given more relays?

Run: python3 -m swarm_sim.relay_k_sweep_all
Outputs: results/relay_k_sweep_all.csv (per-seed rows, written incrementally; the
script resumes from an existing CSV), results/relay_k_sweep_all.png (AUC vs K).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .relay import RelayStrategy
from .relay_baselines import (CentroidRelayStrategy, CentroidOfFarRelayStrategy,
                              FiedlerRelayStrategy)
from .comm import RangeComm
from .metrics import resilience_auc

BASE = dict(width=40, height=40, n_obstacles=25, obstacle_clusters=6,
            n_drones=12, n_victims=12, max_steps=600, seed=300, comm_radius=18.0)
N_SEEDS = 10
RADII = [18, 12, 9, 6, 4]
K_VALUES = [2, 4, 6, 8]

METHODS = {
    "worst_gap": lambda k: RelayStrategy(CentralizedStrategy(), n_relays=k),
    "centroid":  lambda k: CentroidRelayStrategy(CentralizedStrategy(), n_relays=k),
    "centfar":   lambda k: CentroidOfFarRelayStrategy(CentralizedStrategy(), n_relays=k),
    "fiedler":   lambda k: FiedlerRelayStrategy(CentralizedStrategy(), n_relays=k),
}

CSV_FIELDS = ["method", "K", "disk_radius", "seed", "coverage", "detection",
              "connectivity", "mission_time"]

# (method, K) cells: no_relay is K=0 with the plain centralized strategy.
CELLS = [("no_relay", 0)] + [(m, k) for m in METHODS for k in K_VALUES]


def _run_one(method, k, radius, seed_offset):
    """One episode; seeding identical across methods/K for a paired design."""
    cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + seed_offset})
    rng = np.random.default_rng(cfg.seed + 9999)
    comm = RangeComm(radius=radius, rng=rng, mode="disk")
    strat = CentralizedStrategy() if method == "no_relay" else METHODS[method](k)
    m = Simulator(cfg, strat, comm=comm, multihop=True).run()
    return dict(method=method, K=k, disk_radius=radius, seed=cfg.seed,
                coverage=round(m.coverage, 4), detection=round(m.detection_rate, 4),
                connectivity=round(m.mean_connectivity, 4),
                mission_time=m.mission_time)


def _load_done(csv_path):
    done = set()
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((r["method"], int(r["K"]), int(r["disk_radius"]),
                          int(r["seed"])))
    return done, rows


def run_all(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "relay_k_sweep_all.csv")
    done, _ = _load_done(csv_path)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        total = len(CELLS) * len(RADII) * N_SEEDS
        n = len(done)
        for radius in RADII:
            for method, k in CELLS:
                for s in range(N_SEEDS):
                    key = (method, k, radius, BASE["seed"] + s)
                    if key in done:
                        continue
                    row = _run_one(method, k, radius, s)
                    w.writerow(row)
                    f.flush()
                    n += 1
                print(f"[{n}/{total}] radius={radius} {method} K={k} done",
                      flush=True)
    return csv_path


def analyze(out_dir="results"):
    csv_path = os.path.join(out_dir, "relay_k_sweep_all.csv")
    _, rows = _load_done(csv_path)
    # mean coverage per (method, K, radius)
    cov = {}
    for r in rows:
        key = (r["method"], int(r["K"]), int(r["disk_radius"]))
        cov.setdefault(key, []).append(float(r["coverage"]))
    sev = [(RADII[0] - r) / (RADII[0] - RADII[-1]) for r in RADII]

    auc = {}
    curves = {}
    for method, k in CELLS:
        covs = [float(np.mean(cov[(method, k, rad)])) for rad in RADII]
        curves[(method, k)] = covs
        auc[(method, k)] = resilience_auc(sev, covs, baseline=covs[0])

    print("\nFleet=12 (carve-out: searchers = 12 - K), multihop ON, centralized base")
    print("Mean coverage by radius, then resilience AUC "
          "(coverage retention vs radius, normalized by own radius-18 coverage):\n")
    hdr = "method".ljust(11) + "K".rjust(3) + "".join(f"  r={r:<5}" for r in RADII) \
          + "   AUC"
    print(hdr)
    for method, k in CELLS:
        covs = curves[(method, k)]
        print(f"{method:<11}{k:>3}" + "".join(f"  {c:<7.3f}" for c in covs)
              + f"  {auc[(method, k)]:.3f}")

    print("\nRanking of relay methods by AUC at each K "
          "(does more relays change the order?):")
    for k in K_VALUES:
        order = sorted(METHODS, key=lambda m: -auc[(m, k)])
        print(f"  K={k}: " + " > ".join(f"{m}({auc[(m, k)]:.3f})" for m in order))
    print(f"  no_relay reference AUC = {auc[('no_relay', 0)]:.3f}")

    _plot(auc, curves, out_dir)
    return auc, curves


# Okabe-Ito colorblind-safe hues, fixed assignment per method (never cycled),
# with distinct markers as secondary (non-color) encoding.
STYLE = {
    "worst_gap": dict(color="#0072B2", marker="o", ls="-"),
    "centroid":  dict(color="#E69F00", marker="^", ls="-"),
    "centfar":   dict(color="#009E73", marker="d", ls="-"),
    "fiedler":   dict(color="#CC79A7", marker="s", ls="-"),
}


def _plot(auc, curves, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: AUC vs K per method + no-relay reference line
    for m, st in STYLE.items():
        ax1.plot(K_VALUES, [auc[(m, k)] for k in K_VALUES], label=m,
                 lw=2, ms=7, **st)
    ax1.axhline(auc[("no_relay", 0)], color="#555555", ls=":", lw=2)
    ax1.annotate(f"no relay ({auc[('no_relay', 0)]:.2f})",
                 (K_VALUES[0], auc[("no_relay", 0)]),
                 textcoords="offset points", xytext=(2, 5), fontsize=9,
                 color="#555555")
    ax1.set_xlabel("K relays (searchers = 12 - K)")
    ax1.set_ylabel("resilience AUC (coverage retention)")
    ax1.set_title("Resilience AUC vs relay count K")
    ax1.set_xticks(K_VALUES)
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)

    # Right: coverage at the harshest radii vs K (where methods separate)
    for m, st in STYLE.items():
        ax2.plot(K_VALUES, [curves[(m, k)][RADII.index(6)] for k in K_VALUES],
                 lw=2, ms=7, label=f"{m} (r=6)", **st)
        ax2.plot(K_VALUES, [curves[(m, k)][RADII.index(9)] for k in K_VALUES],
                 lw=1.4, ms=5, alpha=0.45, **{**st, "ls": "--"})
    ax2.set_xlabel("K relays (searchers = 12 - K)")
    ax2.set_ylabel("mean coverage")
    ax2.set_title("Coverage vs K at r=6 (solid) and r=9 (dashed)")
    ax2.set_xticks(K_VALUES)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    p = os.path.join(out_dir, "relay_k_sweep_all.png")
    fig.savefig(p, dpi=120)
    print(f"Saved plot: {p}")


def main():
    if "--analyze-only" not in sys.argv:
        run_all()
    analyze()


if __name__ == "__main__":
    main()

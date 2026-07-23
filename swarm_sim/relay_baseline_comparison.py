"""Head-to-head comparison: worst-gap (single-farthest-agent) relay anchoring vs. three
baselines from the connectivity-maintenance literature (centroid, CENTFAR-style
centroid-of-far-nodes, and Fiedler-value/algebraic-connectivity maximization), all
implemented in this simulator and run under identical conditions.

Same map/fleet/seeds/K as relay_experiment.py and relay_k_sweep.py for an apples-to-apples
comparison: 40x40, 12 drones, 4 relays, multihop routing ON, centralized base strategy
(the one relays are shown to rescue), disk-range comm swept from generous to severe.

Run: python -m swarm_sim.relay_baseline_comparison
Outputs: results/relay_baseline_comparison.csv, results/relay_baseline_comparison.png
"""

from __future__ import annotations

import csv
import os

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
N_RELAYS = 4

VARIANTS = {
    "no_relay":       lambda: CentralizedStrategy(),
    "centroid":       lambda: CentroidRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "centfar":        lambda: CentroidOfFarRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "fiedler":        lambda: FiedlerRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "worst_gap":      lambda: RelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
}


def _run(make_strat, radius):
    cov, det, conn = [], [], []
    for s in range(N_SEEDS):
        cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + s})
        rng = np.random.default_rng(cfg.seed + 9999)
        comm = RangeComm(radius=radius, rng=rng, mode="disk")
        m = Simulator(cfg, make_strat(), comm=comm, multihop=True).run()
        cov.append(m.coverage); det.append(m.detection_rate)
        conn.append(m.mean_connectivity)
    return float(np.mean(cov)), float(np.mean(det)), float(np.mean(conn))


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    series = {k: {"cov": [], "det": [], "conn": []} for k in VARIANTS}
    for rad in RADII:
        for name, fac in VARIANTS.items():
            cov, det, conn = _run(fac, rad)
            series[name]["cov"].append(cov)
            series[name]["det"].append(det)
            series[name]["conn"].append(conn)
            rows.append(dict(variant=name, disk_radius=rad,
                             coverage=round(cov, 3), detection=round(det, 3),
                             connectivity=round(conn, 3)))

    with open(os.path.join(out_dir, "relay_baseline_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"40x40, 12 drones, {N_RELAYS} relays, multihop ON, centralized base strategy")
    print("coverage by disk comm radius (smaller = worse)\n")
    print("radius".ljust(8) + "".join(f"{k:>14}" for k in VARIANTS))
    for i, rad in enumerate(RADII):
        print(f"{rad:<8}" + "".join(f"{series[k]['cov'][i]:>14.3f}" for k in VARIANTS))

    # resilience AUC (normalized by each variant's own best/generous-radius coverage)
    print("\nresilience AUC (coverage retention vs radius; 1.0 = flat/ideal):")
    for name in VARIANTS:
        covs = series[name]["cov"]
        # severities = how degraded, 0 (radius=18) .. 1 (radius=4), for AUC's x-axis
        sev = [ (RADII[0] - r) / (RADII[0] - RADII[-1]) for r in RADII ]
        auc = resilience_auc(sev, covs, baseline=covs[0])
        print(f"  {name:<14} AUC={auc:.3f}  (coverage {covs[0]:.3f} -> {covs[-1]:.3f})")

    _plot(series, out_dir)
    return series


def _plot(series, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = RADII
    styles = {"no_relay": ":k", "centroid": "-^", "centfar": "-d",
             "fiedler": "-s", "worst_gap": "--o"}
    for name in VARIANTS:
        style = styles.get(name, "-o")
        ax1.plot(x, series[name]["cov"], style, label=name)
        ax2.plot(x, series[name]["conn"], style, label=name)
    for ax, ttl, yl in ((ax1, "Coverage vs comm range", "coverage"),
                        (ax2, "Connectivity vs comm range", "largest-CC fraction")):
        ax.set_xlabel("disk comm radius (smaller = worse)")
        ax.set_ylabel(yl); ax.set_title(ttl)
        ax.invert_xaxis(); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "relay_baseline_comparison.png")
    fig.savefig(p, dpi=120)
    print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()

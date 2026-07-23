"""Seed expansion + statistical rigor for the relay-anchoring head-to-head.

Reruns the exact relay_baseline_comparison.py setup (40x40 map, 12 drones, K=4 relays,
multihop routing ON, centralized base strategy, disk comm radii [18, 12, 9, 6, 4],
5 variants: no_relay / centroid / centfar / fiedler / worst_gap) but with 40 seeds
instead of 10, then computes:

- per (variant, radius): mean coverage with a 95% t-based confidence interval;
- per variant: per-seed resilience AUC (normalized, per-seed baseline = that seed's
  coverage at the most generous radius, mirroring relay_baseline_comparison's
  normalization) with mean + 95% t-CI, plus the aggregate AUC on the mean-coverage
  curve for direct comparability with the original 10-seed numbers;
- paired Wilcoxon signed-rank tests of worst_gap vs. each other variant, paired per
  seed (identical map/comm seeds across variants), at each radius (on coverage) and
  overall (on per-seed AUC).

Paired design: for a given (radius, seed), every variant sees the same SimConfig seed
(BASE seed + s) and the same comm RNG seed, so per-seed differences isolate the relay
strategy. This is the same seeding scheme as relay_baseline_comparison.py.

Resume-safe: per-run rows are appended to results/relay_seed_expansion.csv as they
complete; on restart, already-completed (variant, radius, seed) runs are skipped.

Run: python -m swarm_sim.relay_seed_expansion
Outputs: results/relay_seed_expansion.csv, results/relay_seed_expansion.png,
         results/relay_seed_expansion_stats.json
"""

from __future__ import annotations

import csv
import json
import math
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

# --- identical config to relay_baseline_comparison.py, except N_SEEDS ---
BASE = dict(width=40, height=40, n_obstacles=25, obstacle_clusters=6,
            n_drones=12, n_victims=12, max_steps=600, seed=300, comm_radius=18.0)
N_SEEDS = 40
RADII = [18, 12, 9, 6, 4]
N_RELAYS = 4

VARIANTS = {
    "no_relay":  lambda: CentralizedStrategy(),
    "centroid":  lambda: CentroidRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "centfar":   lambda: CentroidOfFarRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "fiedler":   lambda: FiedlerRelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "worst_gap": lambda: RelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
}

CSV_FIELDS = ["variant", "disk_radius", "seed", "coverage", "detection", "connectivity"]


def _run_one(variant: str, radius: float, s: int) -> dict:
    cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + s})
    rng = np.random.default_rng(cfg.seed + 9999)
    comm = RangeComm(radius=radius, rng=rng, mode="disk")
    m = Simulator(cfg, VARIANTS[variant](), comm=comm, multihop=True).run()
    return dict(variant=variant, disk_radius=radius, seed=s,
                coverage=float(m.coverage), detection=float(m.detection_rate),
                connectivity=float(m.mean_connectivity))


def _load_done(csv_path: str):
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append(dict(variant=r["variant"], disk_radius=float(r["disk_radius"]),
                                 seed=int(r["seed"]), coverage=float(r["coverage"]),
                                 detection=float(r["detection"]),
                                 connectivity=float(r["connectivity"])))
    return rows


def run_all(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "relay_seed_expansion.csv")
    rows = _load_done(csv_path)
    done = {(r["variant"], r["disk_radius"], r["seed"]) for r in rows}
    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        w.writeheader(); f.flush()
    total = len(RADII) * len(VARIANTS) * N_SEEDS
    for rad in RADII:
        for name in VARIANTS:
            for s in range(N_SEEDS):
                key = (name, float(rad), s)
                if key in done:
                    continue
                r = _run_one(name, float(rad), s)
                rows.append(r)
                w.writerow(r); f.flush()
                done.add(key)
                if len(done) % 25 == 0:
                    print(f"progress: {len(done)}/{total}", flush=True)
    f.close()
    return rows


# ---- statistics -------------------------------------------------------------

def _t_ci(x, conf=0.95):
    """Mean and 95% t-based CI half-width."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return m, 0.0
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.5 + conf / 2.0, n - 1))
    except ImportError:
        tcrit = 2.0227 if n == 40 else 1.96  # t_{0.975, 39}
    return m, tcrit * se


def _wilcoxon(a, b):
    """Paired Wilcoxon signed-rank p-value (two-sided) of a vs b.

    Uses scipy if available; otherwise a normal approximation with tie correction.
    Zero differences are dropped (Wilcoxon convention).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    try:
        from scipy import stats
        return float(stats.wilcoxon(d, alternative="two-sided",
                                    method="auto").pvalue)
    except ImportError:
        n = len(d)
        ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
        # average ranks for ties
        absd = np.abs(d)
        for v in np.unique(absd):
            m = absd == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
        w_plus = ranks[d > 0].sum()
        mu = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        z = (w_plus - mu) / math.sqrt(var)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        return float(min(1.0, p))


def compute_stats(rows):
    # index: [variant][radius] -> coverage array ordered by seed
    cov = {v: {r: np.full(N_SEEDS, np.nan) for r in map(float, RADII)} for v in VARIANTS}
    conn = {v: {r: np.full(N_SEEDS, np.nan) for r in map(float, RADII)} for v in VARIANTS}
    for r in rows:
        cov[r["variant"]][r["disk_radius"]][r["seed"]] = r["coverage"]
        conn[r["variant"]][r["disk_radius"]][r["seed"]] = r["connectivity"]

    sev = [(RADII[0] - r) / (RADII[0] - RADII[-1]) for r in RADII]

    stats_out = {"config": dict(BASE=BASE, n_seeds=N_SEEDS, radii=RADII,
                                n_relays=N_RELAYS, multihop=True,
                                base_strategy="centralized"),
                 "per_radius": {}, "auc": {}, "wilcoxon": {}}

    # per-seed AUC (normalized by that seed's most-generous-radius coverage)
    auc_seed = {}
    for v in VARIANTS:
        aucs = []
        for s in range(N_SEEDS):
            covs = [cov[v][float(r)][s] for r in RADII]
            aucs.append(resilience_auc(sev, covs, baseline=covs[0]))
        auc_seed[v] = np.array(aucs)

    for v in VARIANTS:
        stats_out["per_radius"][v] = {}
        for r in map(float, RADII):
            m, hw = _t_ci(cov[v][r])
            cm, chw = _t_ci(conn[v][r])
            stats_out["per_radius"][v][str(int(r))] = dict(
                mean_coverage=round(m, 4), ci95_halfwidth=round(hw, 4),
                ci95=[round(m - hw, 4), round(m + hw, 4)],
                mean_connectivity=round(cm, 4), connectivity_ci95_halfwidth=round(chw, 4))
        # aggregate AUC on the mean curve (comparable to the original 10-seed report)
        mean_curve = [float(np.mean(cov[v][float(r)])) for r in RADII]
        agg_auc = resilience_auc(sev, mean_curve, baseline=mean_curve[0])
        m, hw = _t_ci(auc_seed[v])
        stats_out["auc"][v] = dict(
            mean_per_seed_auc=round(m, 4), ci95_halfwidth=round(hw, 4),
            ci95=[round(m - hw, 4), round(m + hw, 4)],
            aggregate_auc_mean_curve=round(agg_auc, 4))

    # paired Wilcoxon: worst_gap vs each other variant
    for other in [v for v in VARIANTS if v != "worst_gap"]:
        entry = {"per_radius_coverage": {}, "overall_auc": None}
        for r in map(float, RADII):
            a, b = cov["worst_gap"][r], cov[other][r]
            entry["per_radius_coverage"][str(int(r))] = dict(
                p_value=_wilcoxon(a, b),
                mean_diff=round(float(np.mean(a - b)), 4),
                n_seeds_worst_gap_better=int(np.sum(a > b)),
                n_seeds_tied=int(np.sum(a == b)))
        a, b = auc_seed["worst_gap"], auc_seed[other]
        entry["overall_auc"] = dict(
            p_value=_wilcoxon(a, b),
            mean_diff=round(float(np.mean(a - b)), 4),
            n_seeds_worst_gap_better=int(np.sum(a > b)))
        stats_out["wilcoxon"][f"worst_gap_vs_{other}"] = entry

    return stats_out, cov, conn, auc_seed


# ---- plot -------------------------------------------------------------------

def _plot(cov, conn, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    styles = {"no_relay": (":", "k", None), "centroid": ("-", "C0", "^"),
              "centfar": ("-", "C1", "d"), "fiedler": ("-", "C2", "s"),
              "worst_gap": ("--", "C3", "o")}
    x = RADII
    for name in VARIANTS:
        ls, c, mk = styles[name]
        for ax, data in ((ax1, cov), (ax2, conn)):
            means, los, his = [], [], []
            for r in x:
                m, hw = _t_ci(data[name][float(r)])
                means.append(m); los.append(m - hw); his.append(m + hw)
            ax.plot(x, means, ls, color=c, marker=mk, label=name)
            ax.fill_between(x, los, his, color=c, alpha=0.18, linewidth=0)
    for ax, ttl, yl in ((ax1, f"Coverage vs comm range ({N_SEEDS} seeds, 95% CI)", "coverage"),
                        (ax2, "Connectivity vs comm range (95% CI)", "largest-CC fraction")):
        ax.set_xlabel("disk comm radius (smaller = worse)")
        ax.set_ylabel(yl); ax.set_title(ttl)
        ax.invert_xaxis(); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "relay_seed_expansion.png")
    fig.savefig(p, dpi=120)
    print(f"Saved plot: {p}")


def main(out_dir="results"):
    rows = run_all(out_dir)
    stats_out, cov, conn, auc_seed = compute_stats(rows)

    print(f"\n40x40, 12 drones, {N_RELAYS} relays, multihop ON, centralized base, "
          f"{N_SEEDS} seeds")
    print("mean coverage (95% CI half-width) by disk comm radius\n")
    print("radius".ljust(8) + "".join(f"{k:>22}" for k in VARIANTS))
    for rad in RADII:
        line = f"{rad:<8}"
        for k in VARIANTS:
            m, hw = _t_ci(cov[k][float(rad)])
            line += f"{m:>13.3f} ±{hw:.3f}"
        print(line)

    print("\nresilience AUC (per-seed mean ± 95% CI; aggregate on mean curve):")
    for v in VARIANTS:
        a = stats_out["auc"][v]
        print(f"  {v:<12} {a['mean_per_seed_auc']:.3f} ±{a['ci95_halfwidth']:.3f}"
              f"  (aggregate {a['aggregate_auc_mean_curve']:.3f})")

    print("\npaired Wilcoxon (worst_gap vs other), two-sided p-values:")
    for k, e in stats_out["wilcoxon"].items():
        pr = "  ".join(f"r{r}: {d['p_value']:.2g}" for r, d in
                       e["per_radius_coverage"].items())
        print(f"  {k:<24} AUC p={e['overall_auc']['p_value']:.2g}   {pr}")

    with open(os.path.join(out_dir, "relay_seed_expansion_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Saved stats: {os.path.join(out_dir, 'relay_seed_expansion_stats.json')}")

    _plot(cov, conn, out_dir)
    return stats_out


if __name__ == "__main__":
    main()

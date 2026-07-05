"""Full experiment matrix: 3 strategies x failure scenarios x seeds.

Scenarios:
  S0  perfect
  S1  packet loss (p = 0.05, 0.10, 0.20)
  S2  range degradation (comm radius shrunk to {0.6,0.4,0.2}x nominal, path-loss)
  S3  dead zone (central rectangle, area fractions {0.1,0.2,0.3})
  S5  node failure: kill 50% of drones at step 150
  S6  combined "realistic disaster": range(0.4x) + dead zone(0.15) + packet loss(0.1)

Run: python -m swarm_sim.full_matrix
Outputs to results/: matrix.csv, matrix_summary.json, degradation_<type>.png, winners.json
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .decentralized import DecentralizedStrategy
from .stigmergy import StigmergyStrategy
from .comm import (PerfectComm, PacketLoss, RangeComm, DeadZone, Composite)
from .metrics import resilience_auc

STRATEGIES = {
    "centralized": CentralizedStrategy,
    "decentralized": DecentralizedStrategy,
    "stigmergy": StigmergyStrategy,
}

N_SEEDS = 10
BASE = SimConfig(n_drones=10, n_victims=15, max_steps=600, seed=100)

# Node failure needs a harder map so search throughput actually matters — on the
# default map the survivors trivially finish, masking any effect.
HARD = dict(width=100, height=100, n_obstacles=160, obstacle_clusters=20,
            n_victims=30, n_drones=12, sensing_radius=3.0, comm_radius=15.0,
            max_steps=800, seed=200)
NODE_FAILURE_STEP = 150   # fires mid-search (t=150; full-fleet convergence ~step 395)


def _dead_zones(area_fraction, cfg):
    """One centered square dead zone covering `area_fraction` of the map."""
    area = area_fraction * cfg.width * cfg.height
    side = int(round(area ** 0.5))
    cy, cx = cfg.height // 2, cfg.width // 2
    h = side // 2
    return [(cy - h, cx - h, cy + h, cx + h)]


def cfg_override(scenario):
    """Per-scenario base-config overrides (node failure uses the harder map)."""
    if scenario in ("node_failure", "node_failure_targeted"):
        return HARD
    return {}


def build(scenario, level, rng, cfg):
    """Return (comm, sim_kwargs) for a scenario at a severity `level`."""
    if scenario == "perfect":
        return PerfectComm(), {}
    if scenario == "packet_loss":
        return PacketLoss(p=level, rng=rng), {}
    if scenario == "range":
        # level = fraction-degraded; radius multiplier = (1 - level)
        return RangeComm(radius=max(1.0, cfg.comm_radius * (1.0 - level)),
                         rng=rng, mode="pathloss"), {}
    if scenario == "dead_zone":
        return DeadZone(_dead_zones(level, cfg)), {}
    if scenario in ("node_failure", "node_failure_targeted"):
        # level = fraction of drones killed mid-search; perfect comms isolates the loss
        n_kill = int(round(level * cfg.n_drones))
        mode = "targeted" if scenario == "node_failure_targeted" else "random"
        return PerfectComm(), {"scheduled_failures": {NODE_FAILURE_STEP: n_kill},
                               "failure_mode": mode}
    if scenario == "combined":
        comm = Composite([
            RangeComm(radius=max(1.0, cfg.comm_radius * 0.4), rng=rng, mode="pathloss"),
            DeadZone(_dead_zones(0.15, cfg)),
            PacketLoss(p=0.10, rng=rng),
        ])
        return comm, {"scheduled_failures": {150: int(0.2 * cfg.n_drones)}}
    raise ValueError(scenario)


# scenario -> list of severity levels (0.0 = the no-failure reference point)
SCENARIOS = {
    "perfect":      [0.0],
    "packet_loss":  [0.0, 0.05, 0.10, 0.20],
    "range":        [0.0, 0.4, 0.6, 0.8],          # fraction the comm radius is cut
    "dead_zone":    [0.0, 0.10, 0.20, 0.30],       # map-area fraction blacked out
    "node_failure":          [0.0, 0.25, 0.50, 0.75],   # random kill, hard map, t=30
    "node_failure_targeted": [0.0, 0.25, 0.50, 0.75],   # kill best-connected hubs
    "combined":     [0.0, 1.0],                    # off / on (fixed realistic mix)
}


def run_cell(factory, scenario, level):
    rows = []
    ov = cfg_override(scenario)
    base = {**BASE.__dict__, **ov}
    for s in range(N_SEEDS):
        cfg = SimConfig(**{**base, "seed": base["seed"] + s})
        rng = np.random.default_rng(cfg.seed + 9999)
        comm, kw = build(scenario, level, rng, cfg)
        m = Simulator(cfg, factory(), comm=comm, **kw).run()
        d = m.as_dict()
        d.update(scenario=scenario, level=level, seed=cfg.seed)
        rows.append(d)
    return rows


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    summary = defaultdict(dict)

    for scenario, levels in SCENARIOS.items():
        for sname, factory in STRATEGIES.items():
            mean_cov, mean_det, mean_mt, lv_used = [], [], [], []
            for lv in levels:
                rows = run_cell(factory, scenario, lv)
                all_rows.extend(rows)
                cov = np.mean([r["coverage"] for r in rows])
                det = np.mean([r["detection_rate"] for r in rows])
                conv = [r["mission_time"] for r in rows if r["mission_time"] >= 0]
                mean_cov.append(float(cov))
                mean_det.append(float(det))
                mean_mt.append(float(np.mean(conv)) if conv else float("nan"))
                lv_used.append(lv)
            base_cov = mean_cov[0] if mean_cov else 1.0
            summary[scenario][sname] = {
                "levels": lv_used,
                "coverage_mean": mean_cov,
                "detection_mean": mean_det,
                "mission_time_mean": mean_mt,
                "coverage_auc": resilience_auc(lv_used, mean_cov, baseline=base_cov)
                                if len(lv_used) > 1 else mean_cov[0],
            }

    _write(all_rows, summary, out_dir)
    _print(summary)
    winners = _winners(summary)
    with open(os.path.join(out_dir, "winners.json"), "w") as f:
        json.dump(winners, f, indent=2)
    _plot(summary, out_dir)
    return summary, winners


def _write(all_rows, summary, out_dir):
    import csv
    keys = list(all_rows[0].keys())
    with open(os.path.join(out_dir, "matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    with open(os.path.join(out_dir, "matrix_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _print(summary):
    for scenario, by in summary.items():
        print(f"\n=== {scenario} : coverage @ severity (detection in parens) ===")
        levels = next(iter(by.values()))["levels"]
        print("severity".ljust(12) + "".join(f"{s:>16}" for s in by))
        for i, lv in enumerate(levels):
            cells = "".join(
                f"{by[s]['coverage_mean'][i]:>8.3f}({by[s]['detection_mean'][i]:>4.2f})"
                for s in by)
            print(f"{lv:<12.2f}{cells}")
        print("cov AUC".ljust(12) + "".join(f"{by[s]['coverage_auc']:>16.3f}" for s in by))


def _winners(summary):
    """Pick the best strategy per scenario by coverage AUC (worst severity tiebreak)."""
    out = {}
    for scenario, by in summary.items():
        ranked = sorted(by.items(),
                        key=lambda kv: (kv[1]["coverage_auc"],
                                        kv[1]["coverage_mean"][-1]),
                        reverse=True)
        out[scenario] = {
            "winner": ranked[0][0],
            "ranking": [{"strategy": k, "coverage_auc": round(v["coverage_auc"], 3),
                         "coverage_at_worst": round(v["coverage_mean"][-1], 3)}
                        for k, v in ranked],
        }
    return out


def _plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plots skipped: {e})")
        return
    for scenario, by in summary.items():
        if len(next(iter(by.values()))["levels"]) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for sname, d in by.items():
            ax.plot(d["levels"], d["coverage_mean"], marker="o", label=sname)
        ax.set_xlabel(f"{scenario} severity")
        ax.set_ylabel("coverage fraction")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Coverage degradation — {scenario}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"degradation_{scenario}.png")
        fig.savefig(p, dpi=120)
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()

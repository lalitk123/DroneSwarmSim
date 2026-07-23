"""EXPERIMENT 5 — Continuous re-anchoring under attrition.

The claim element distinguishing worst-gap relay anchoring from one-time placement
prior art (arXiv:2512.09682) is CONTINUOUS per-timestep re-anchoring. This experiment
tests whether that matters: a STATIC one-shot variant computes the worst-gap relay
chain once (at freeze_step, after drones have dispersed) and never repositions the
relays afterwards, while the continuous variant re-anchors every step.

Attrition scenarios (kill applied identically across all methods, paired seeds):
  none              — control, no attrition.
  kill_anchor       — at step 100, kill the farthest living SEARCHER from base (for the
                      continuous method this is exactly the drone the chain is anchored on).
  kill_relay        — at step 100, kill the mid-chain RELAY drone (fixed id = the relay at
                      chain fraction 3/5; for no_relay the same id is a searcher, keeping
                      fleet attrition identical across methods).
  kill_anchor_early / kill_relay_early — same kills at step 40. Added after the step-100
                      runs revealed that static one-shot placement STALLS by step ~50
                      (the frozen chain goes stale as searchers move on), so a step-100
                      kill lands on an already-failed static trajectory and is a no-op;
                      step 40 is inside static's brief live window (freeze at 30) and
                      tests attrition while the frozen chain is still doing work.

Config matches relay_baseline_comparison.py (40x40, 12 drones, K=4 relays,
multihop=True, centralized base strategy, disk comm), radii [12, 9, 6], 10 seeds.
Two freeze times are run for the static variant: t=0 (spawn positions — the literal
"place at mission start" prior art) and t=30 (after dispersal — the strongest fair
static baseline).

Run: /usr/bin/python3 -m swarm_sim.relay_attrition
Outputs: results/relay_attrition.csv (per-run rows, resumable),
         results/relay_attrition.png
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy, CoordinationStrategy
from .relay import RelayStrategy
from .comm import RangeComm

BASE = dict(width=40, height=40, n_obstacles=25, obstacle_clusters=6,
            n_drones=12, n_victims=12, max_steps=600, seed=300, comm_radius=18.0)
N_SEEDS = 10
RADII = [12, 9, 6]
N_RELAYS = 4
KILL_STEP = 100
N_SEARCHERS = BASE["n_drones"] - N_RELAYS          # ids 0..7 search
RELAY_IDS = list(range(N_SEARCHERS, BASE["n_drones"]))  # ids 8..11 relay
# mid-chain relay: relays are enumerated in id order with chain fraction (i+1)/(K+1);
# index K//2 = 2 -> id 10 sits at fraction 3/5 of the base->anchor chain.
MID_RELAY_ID = RELAY_IDS[N_RELAYS // 2]

CSV_PATH = os.path.join("results", "relay_attrition.csv")
PNG_PATH = os.path.join("results", "relay_attrition.png")
FIELDS = ["scenario", "method", "radius", "seed", "coverage", "detection",
          "connectivity", "mission_time", "n_alive_end"]


class StaticRelayStrategy(RelayStrategy):
    """One-shot worst-gap relay placement (prior-art model).

    Identical to RelayStrategy (same placement geometry, inherited) until
    `freeze_step`: relays chase the current worst-gap chain while the swarm
    disperses. At `freeze_step` the chain targets are recorded and FROZEN —
    relays hold those positions forever, regardless of searcher motion or
    attrition. freeze_step=0 freezes the chain computed from spawn positions
    (literal placement-at-mission-start)."""

    def __init__(self, base_strategy, n_relays=3, freeze_step=30):
        super().__init__(base_strategy, n_relays)
        self.freeze_step = int(freeze_step)
        self.name = f"static_{self.freeze_step}({base_strategy.name},k={n_relays})"
        self._frozen = None

    def reset(self, world, drones, base):
        self._frozen = None
        super().reset(world, drones, base)

    def assign(self, world, drones, base, send, step):
        if self._frozen is None:
            # continuous (parent) behavior up to and including the freeze step
            super().assign(world, drones, base, send, step)
            if step >= self.freeze_step:
                self._frozen = {d.id: d.target for d in drones
                                if d.id in self._relay_ids and d.target is not None}
            return
        # after the freeze: searchers still run the wrapped strategy normally,
        # relays hold the frozen chain positions.
        searchers = [d for d in drones if d.id not in self._relay_ids]
        self.base.assign(world, searchers, base, send, step)
        for d in drones:
            if d.id in self._relay_ids and d.alive:
                d.target = self._frozen.get(d.id, d.target)


class KillWrapper(CoordinationStrategy):
    """Applies the scenario's attrition at KILL_STEP, before delegating.

    Simulator's built-in scheduled_failures kills random or highest-degree drones —
    neither targets the chain anchor — so the kill is done here, identically for
    every method (non-invasive: no existing file modified).

    kind='anchor': kill the farthest living searcher-id drone from base.
    kind='relay' : kill the drone with id MID_RELAY_ID (a mid-chain relay for the
                   relay methods; the same id for no_relay so fleet size stays paired).
    kind=None    : control, no kill."""

    def __init__(self, inner: CoordinationStrategy, kind, kill_step):
        self.inner = inner
        self.kind = kind
        self.kill_step = kill_step
        self.name = f"kill[{kind}]({inner.name})"
        self._done = False

    def reset(self, world, drones, base):
        self._done = False
        self.inner.reset(world, drones, base)

    def _pick(self, drones, base):
        if self.kind == "anchor":
            cands = [d for d in drones if d.alive and d.id < N_SEARCHERS]
            if not cands:
                return None
            dists = [np.linalg.norm(d.pos - base.pos) for d in cands]
            return cands[int(np.argmax(dists))]
        if self.kind == "relay":
            for d in drones:
                if d.id == MID_RELAY_ID and d.alive:
                    return d
        return None

    def assign(self, world, drones, base, send, step):
        if not self._done and self.kind is not None and step >= self.kill_step:
            victim = self._pick(drones, base)
            if victim is not None:
                victim.alive = False
                victim.target = None
            self._done = True
        self.inner.assign(world, drones, base, send, step)


METHODS = {
    "no_relay":   lambda: CentralizedStrategy(),
    "static_t0":  lambda: StaticRelayStrategy(CentralizedStrategy(),
                                              n_relays=N_RELAYS, freeze_step=0),
    "static_t30": lambda: StaticRelayStrategy(CentralizedStrategy(),
                                              n_relays=N_RELAYS, freeze_step=30),
    "worst_gap":  lambda: RelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
}
# scenario -> (kill kind, kill step)
SCENARIOS = {"none": (None, KILL_STEP),
             "kill_anchor": ("anchor", KILL_STEP),
             "kill_relay": ("relay", KILL_STEP),
             "kill_anchor_early": ("anchor", 40),
             "kill_relay_early": ("relay", 40)}


def _run_one(method, scenario, radius, seed_off):
    cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + seed_off})
    rng = np.random.default_rng(cfg.seed + 9999)
    comm = RangeComm(radius=radius, rng=rng, mode="disk")
    kind, kstep = SCENARIOS[scenario]
    strat = KillWrapper(METHODS[method](), kind, kstep)
    m = Simulator(cfg, strat, comm=comm, multihop=True).run()
    return dict(scenario=scenario, method=method, radius=radius,
                seed=cfg.seed, coverage=round(m.coverage, 4),
                detection=round(m.detection_rate, 4),
                connectivity=round(m.mean_connectivity, 4),
                mission_time=m.mission_time, n_alive_end=m.n_alive_end)


def _load_done():
    done = set()
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for r in csv.DictReader(f):
                done.add((r["scenario"], r["method"], int(r["radius"]),
                          int(r["seed"])))
                rows.append(r)
    return done, rows


def main():
    os.makedirs("results", exist_ok=True)
    done, rows = _load_done()
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for scenario in SCENARIOS:
            for radius in RADII:
                for method in METHODS:
                    for s in range(N_SEEDS):
                        key = (scenario, method, radius, BASE["seed"] + s)
                        if key in done:
                            continue
                        row = _run_one(method, scenario, radius, s)
                        rows.append({k: str(v) for k, v in row.items()})
                        w.writerow(row)
                        f.flush()
                print(f"done: {scenario} r={radius}", flush=True)
    summarize(rows)
    plot(rows)


def _mean_cov(rows, scenario, method, radius):
    vs = [float(r["coverage"]) for r in rows
          if r["scenario"] == scenario and r["method"] == method
          and int(r["radius"]) == radius]
    return float(np.mean(vs)) if vs else float("nan")


def _paired_delta(rows, scenario, a, b, radius):
    """Per-seed coverage delta a - b (paired by seed)."""
    da = {int(r["seed"]): float(r["coverage"]) for r in rows
          if r["scenario"] == scenario and r["method"] == a
          and int(r["radius"]) == radius}
    db = {int(r["seed"]): float(r["coverage"]) for r in rows
          if r["scenario"] == scenario and r["method"] == b
          and int(r["radius"]) == radius}
    return [da[s] - db[s] for s in sorted(set(da) & set(db))]


def summarize(rows):
    print("\ncoverage (mean over seeds) by scenario / radius:")
    hdr = "scenario".ljust(13) + "radius".ljust(8) + "".join(f"{m:>12}" for m in METHODS)
    print(hdr)
    for scenario in SCENARIOS:
        for radius in RADII:
            line = f"{scenario:<13}{radius:<8}"
            line += "".join(f"{_mean_cov(rows, scenario, m, radius):>12.3f}"
                            for m in METHODS)
            print(line)
    print("\ncontinuous-minus-static coverage delta (paired by seed, mean +/- sd):")
    for scenario in SCENARIOS:
        for static in ("static_t0", "static_t30"):
            for radius in RADII:
                d = _paired_delta(rows, scenario, "worst_gap", static, radius)
                if d:
                    print(f"  {scenario:<13} vs {static:<11} r={radius:<3} "
                          f"delta={np.mean(d):+.3f} +/- {np.std(d):.3f} "
                          f"(n={len(d)}, {sum(x > 0 for x in d)} seeds favor continuous)")


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference categorical palette, fixed slot order (validated set)
    colors = {"no_relay": "#2a78d6", "static_t0": "#1baf7a",
              "static_t30": "#eda100", "worst_gap": "#008300"}
    markers = {"no_relay": "x", "static_t0": "^", "static_t30": "s",
               "worst_gap": "o"}
    ls = {"no_relay": ":", "static_t0": "--", "static_t30": "--",
          "worst_gap": "-"}
    titles = {"none": "no attrition (control)",
              "kill_anchor": "anchor searcher killed @ step 100",
              "kill_relay": "mid-chain relay killed @ step 100",
              "kill_anchor_early": "anchor searcher killed @ step 40",
              "kill_relay_early": "mid-chain relay killed @ step 40"}

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.3 * len(SCENARIOS), 4.4),
                             sharey=True, facecolor="#fcfcfb")
    for ax, scenario in zip(axes, SCENARIOS):
        ax.set_facecolor("#fcfcfb")
        for m in METHODS:
            ys = [_mean_cov(rows, scenario, m, r) for r in RADII]
            ax.plot(RADII, ys, ls[m], color=colors[m], marker=markers[m],
                    markersize=7, linewidth=2, label=m)
        ax.set_title(titles[scenario], fontsize=11, color="#0b0b0b")
        ax.set_xlabel("disk comm radius (smaller = worse)", color="#52514e")
        ax.invert_xaxis()
        ax.set_xticks(RADII)
        ax.set_ylim(0, 1.05)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.tick_params(colors="#898781")
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
    axes[0].set_ylabel("coverage at mission end", color="#52514e")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.suptitle("Continuous worst-gap re-anchoring vs static one-shot placement "
                 "under attrition (40x40, 12 drones, K=4, 10 seeds)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PNG_PATH, dpi=120)
    print(f"\nSaved plot: {PNG_PATH}")


if __name__ == "__main__":
    main()

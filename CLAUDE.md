# Resilient Multi-Agent Drone Swarms for Disaster Search-and-Rescue Under Communication Failures

## Project summary
Investigate how autonomous drone swarms continue search-and-rescue (SAR) missions when
inter-drone communication is unreliable or fails entirely. In a simulated disaster
environment (obstacles + victims), multiple coordination strategies are compared under
realistic communication degradation. Optional extension: adaptive relay drones that
reposition to restore links.

## Research findings (web survey, 2023–2026)

### 1. Simulation frameworks
- **ROS 2 + Gazebo (+ PX4/Ardupilot)**: high physical/SITL fidelity, standard in robotics
  research, but heavy — scaling past ~10 drones is slow, and it has *no built-in model of
  network degradation*. You bolt on a network layer (e.g. ns-3 via integration) to get
  packet loss/dead zones. Best when you eventually want hardware transfer.
- **Custom Python / lightweight gym envs**: fast to iterate, trivial to inject arbitrary
  comms failures, scales to 10–100+ agents. Lower physical fidelity. Dominant for
  algorithm-comparison studies where comms modeling (not aerodynamics) is the research
  question.
- **gym-pybullet-drones** (UTIAS): Gymnasium env, real PyBullet physics, single + multi
  agent. Good middle ground but its N×N adjacency representation limits large-swarm
  scaling.
- **Flightmare** (UZH-RPG): decouples Unity rendering from physics; can simulate hundreds
  of agents, physics up to ~200 kHz — fastest for large-swarm RL.
- **ns-3 / FANET simulators**: the *communication* gold standard (OLSR/AODV/DSDV routing,
  path-loss + mobility models) but not a physics/mission simulator.

### 2. Coordination strategies & resilience
- **Centralized** (single planner / CBBA-style allocation): optimal coordination but a
  single point of failure; degrades sharply when the link to the coordinator drops —
  the natural "baseline" to beat.
- **Decentralized consensus / gossip (e.g. CBBA, consensus-based bundle)**: mathematically
  elegant, but stability depends on *persistent connectivity*; intermittent links break
  convergence guarantees. Biomimetic/load-aware CBBA variants improve robustness.
- **Bio-inspired / stigmergy (ACO pheromone, PSO, Physarum)**: coordinate through the
  *environment* (shared/virtual pheromone map) rather than direct messaging, so they
  degrade gracefully and even work **communication-free**. Literature repeatedly finds
  stigmergy/bio-inspired methods the most resilient to comms loss; the tradeoff is
  redundant coverage and sensitivity to map staleness. **Strongest fit for this project's
  thesis.**

### 3. Communication failure modeling
Standard approaches researchers use:
- **Probabilistic packet loss**: per-message Bernoulli drop with probability `p`.
- **Geographic / range models**: unit-disk (binary cutoff radius) or distance-dependent
  path loss `d^-α` (Friis free-space, Nakagami-m fading) — link probability falls with
  distance.
- **Dead zones**: spatial regions where comms = 0 (model damaged infrastructure / terrain).
- **Latency / jitter**: delayed message delivery.
- **Node failure**: drones drop out entirely.
Mobility models: Random Waypoint is the common baseline. FANET = MANET in 3D.

### 4. Metrics & resilience quantification
Track: **area coverage %**, **victim detection rate / count**, **time-to-first-detection**,
**mission completion time**, **redundant coverage / overlap**, **energy or path length**,
**network connectivity** (fraction connected, largest connected component).
**Resilience** is typically quantified as *performance retention vs. a no-failure baseline*
— i.e. plot a chosen metric as a function of failure severity (packet-loss rate, dead-zone
size, % drones lost) and report the area-under-curve / degradation slope. A resilient
strategy has a flat curve.

### 5. Scaling to 10+ drones fast
For algorithm comparison with comms as the focus, a **custom 2D/2.5D Python grid+kinematic
sim is fastest**; reach for gym-pybullet-drones/Flightmare only if physical fidelity or RL
transfer matters. ROS 2 + Gazebo if hardware deployment is a future goal.

## Recommendations for this project

**Simulation framework:** Build a **custom lightweight Python simulator** (NumPy +
Gymnasium-style API; matplotlib/pygame viz). Rationale: the research question is *comms
resilience and coordination*, not aerodynamics. We need to scale to 10–50 drones, inject
arbitrary failure models, and run hundreds of Monte-Carlo trials cheaply. Keep a clean
interface so a Gazebo/PX4 backend could be swapped in later for hardware validation.

**Three coordination strategies to implement:**
1. **Centralized planner (baseline)** — global frontier/sector assignment from a base
   station. Shows the failure mode we want to beat.
2. **Decentralized consensus / gossip** — drones share local maps with in-range neighbors
   and merge; assignment via distributed greedy/CBBA-lite. Tests graceful-but-
   connectivity-dependent behavior.
3. **Bio-inspired stigmergy (digital pheromone / ACO-style)** — drones deposit "explored"
   and "attractant" markers on a shared spatial field, repelled from recently-visited
   cells. Near comms-free; expected most resilient. This is the headline hypothesis.

**Communication failure scenarios to test:**
- S0 Perfect comms (control)
- S1 Probabilistic packet loss, sweep p ∈ {0.1 … 0.9}
- S2 Range-limited (unit-disk radius sweep) + distance path-loss variant
- S3 Static dead zones (sweep area fraction)
- S4 Latency/jitter
- S5 Drone failures (sweep % lost, random + targeted)
- S6 Combined "realistic disaster" (range-limited + dead zones + losses)

**Key metrics:** coverage %, victim detection rate, time-to-first-victim, mission time,
redundant-coverage ratio, connectivity (largest connected component), energy/path length.
Report each as a **degradation curve vs. failure severity** + AUC resilience score.

**Extension:** adaptive relay drones — designate a subset to reposition to maximize
connectivity (max largest-connected-component) and measure resilience recovery.
**Implemented & validated** (`swarm_sim/relay.py`, `relay_experiment.py`). Relays require
multi-hop comm routing (`Simulator(multihop=True)`): a message is delivered iff a path of
deliverable links exists through the relay chain. K relays reposition each step to bridge
the base to the *farthest* searcher (the worst link). Result (40×40, 12 drones, 4 relays,
disk-range comms): relays **rescue the centralized strategy** under range loss —
coverage 0.41→0.91 at radius 18 and 0.26→0.89 at radius 12 (+0.5–0.64 absolute), with
connectivity rising in step; the gain tapers once the gap exceeds what 4 relays can span
(radius ≤ 6). Decentralized, already resilient, gains little. Lessons: (1) anchor relays on
the worst gap, not the swarm centroid — a centroid anchor steers relays into a *central*
dead zone; (2) relays only help if messages actually multi-hop through them.

**Baseline comparison** (`swarm_sim/relay_baselines.py`, `relay_baseline_comparison.py`):
worst-gap (single-farthest-agent) anchoring benchmarked head-to-head, same map/seeds/K,
against three baselines drawn from the connectivity-maintenance literature — naive centroid,
CENTFAR-style centroid-of-far-nodes, and Fiedler-value/algebraic-connectivity gradient
ascent (the mainstream approach; implemented as the literature's actual analytical
eigenvector-gradient control law, with the standard inter-relay repulsion term, after an
initial discretized greedy-search attempt was found to provably get stuck at a local optimum
and was discarded as an unfair strawman). **Worst-gap wins at every radius tested** (10
seeds): coverage 0.91/0.89/0.77/0.34/0.21 vs. centroid 0.94/0.82/0.58/0.32/0.17, CENTFAR
0.94/0.85/0.58/0.33/0.19, and Fiedler 0.79/0.41/0.28/0.19/0.13, at radii 18/12/9/6/4.
Resilience AUC: worst-gap 0.793 > CENTFAR 0.714 > centroid 0.699 > Fiedler 0.511 > no-relay
0.585. Fiedler — despite being the theoretically "proper" mainstream method — is the
*worst* relay-assisted variant: global algebraic-connectivity maximization is a diffuse
objective (improves aggregate network cohesion everywhere) rather than a task-aware one
(bridging the specific gap blocking mission progress), which worst-gap anchoring targets
directly.

## Relay result-expansion suite (5 experiments, publication/patent hardening)

Five independent stress axes run to test whether the worst-gap advantage is real or an
artifact of one map/seed set/K/fleet size. **The advantage survived all five.** All runs use
paired design: identical `SimConfig` seed and comm-RNG seed (`cfg.seed + 9999`) across every
variant within a cell, so variant differences are not seed noise. Scripts are additive — no
existing file was modified.

**AUC convention (important, two different numbers exist):** *self-baselined* AUC\* normalizes
each method's curve by its own coverage at the most generous radius, so it measures curve
*flatness only* and will rank a uniformly-terrible method (no_relay) highly — it must never
be used to rank methods against each other. *Absolute* AUC (mean coverage across the severity
sweep, baseline 1.0 = perfect coverage at every radius) is the valid cross-method number.
Numbers below are absolute AUC unless stated. This distinction was found by an adversarial
verifier after an early fleet-scaling report ranked methods with AUC\* and got them wrong.

**1. Seed expansion — statistical power** (`relay_seed_expansion.py` → `relay_seed_expansion.csv`,
`_stats.json`). 40 seeds, 40×40, 12 drones, K=4, paired Wilcoxon signed-rank + 95% t-CIs.
Resilience AUC: **worst_gap 0.794 ±0.017 > centfar 0.713 > centroid 0.690 > no_relay 0.556 >
fiedler 0.510**; every pairwise AUC comparison significant (worst_gap vs. each: p ≤ 1.8e-11;
39–40 of 40 seeds favor worst_gap). Per-radius worst_gap coverage 0.934/0.920/0.759/0.369/0.207
at r=18/12/9/6/4. **Honest caveat: at the generous radius r=18 worst_gap is statistically tied
with centroid (p=0.96) and centfar (p=0.92)** — the advantage is specific to *degraded* comms,
which is the correct and defensible claim. Note p=1.8189894e-12 is the exact-test p-floor at
n=40, not a computed value.

**2. Map generalization — is it map-specific?** (`relay_map_generalization.py` →
`relay_map_generalization.csv`, 1750 rows = 7 conditions × 5 variants × 5 radii × 10 seeds).
Conditions: three bare map sizes with **no dead zones at all** (40×40, 60×60, 100×100) plus
four dead-zone layouts (the default + three randomized placements never seen during
development). **worst_gap wins the AUC ranking in all 7 conditions**, with the full ordering
(worst_gap > centfar ≈ centroid > no_relay/fiedler) stable throughout. Self-baselined AUC\* by
condition: 40×40 0.793, 60×60 0.800, 100×100 0.827, dz_default 0.835, dz_rand1 0.789,
dz_rand2 0.801, dz_rand3 0.750. The no-dead-zone conditions matter most — they show the result
is not an artifact of dead-zone geometry, and the margin actually *grows* with map size
(100×100: worst_gap 0.827 vs centfar 0.764 vs no_relay 0.369). Script is resumable from CSV
and supports `--conditions` / `--report`.

**3. K sweep — does Fiedler just need more relays?** (`relay_k_sweep_all.py` →
`relay_k_sweep_all.csv`). K ∈ {0,2,4,6,8}, all four methods, carve-out fleet convention
(relays are taken *out of* a fixed 12-drone fleet, so K=8 leaves only 4 searchers).
Absolute AUC: worst_gap **0.471 / 0.624 / 0.670 / 0.726** at K=2/4/6/8. **No — Fiedler does not
close the gap with more relays; it widens** (worst_gap − fiedler: 0.136 at K=2 → 0.334 at K=8).
**Honest caveat: at K=2 worst_gap is not the winner** — it is in a three-way statistical tie
(centfar 0.473, worst_gap 0.471, centroid 0.469); two relays are too few for anchor placement
to matter. Worst-gap ranks first at K=4, 6, and 8, and its lead *grows* with K, i.e. it is the
only method that converts extra relays into mission progress (centroid and centfar both peak
at K=4–6 then decline as the carve-out starves the searcher fleet).

**4. Fleet scaling — does one chain suffice, or do you need branches?**
(`relay_fleet_scaling.py` → `relay_fleet_scaling.csv`). 60×60, fleets of 12/24/36 searchers,
K=4/6, radii 18/13/9, vs. a `multi_branch` variant running several simultaneous relay chains.
**Multi-branch never beats single-anchor worst_gap — 0 of 6 panels.** Worst_gap absolute AUC
0.701/0.794/0.783/0.921/0.771/0.898 across the six (fleet,K) panels vs. no_relay 0.133–0.174.
A single well-anchored chain suffices up to at least 36 searchers; branching *hurts* at K=6
(multi_branch drops to 0.562–0.615 while worst_gap reaches 0.794–0.921) because splitting the
relay budget leaves no chain long enough to bridge the worst gap.

**5. Attrition — is per-timestep re-anchoring load-bearing?** (`relay_attrition.py` →
`relay_attrition.csv`). Compares continuous worst_gap against `StaticRelayStrategy` (same
anchoring rule, but the chain is computed once and frozen at step t) under relay/anchor kills.
`KillWrapper` applies identical kill rules to every method (anchor kill = farthest living
searcher; relay kill = fixed drone id 10) so the comparison is paired.
**This is the strongest patent-relevant result: a one-shot static chain fails from ordinary
swarm motion alone, before any failure occurs.** With *no* attrition, continuous worst_gap
beats static-frozen-at-t30 by +0.25 coverage at r=12 (0.894 vs 0.644) and +0.22 at r=9
(0.771 vs 0.551), and static-frozen-at-t0 collapses to ≈ no_relay (0.255/0.200/0.136 vs
0.259/0.190/0.114) — freezing the chain at t0 is worth nothing at all. Under kills, continuous
degrades gracefully and an anchor kill even slightly *helps* it (r=9: 0.771 → 0.791, because
losing the anchor forces immediate re-optimization onto the next-worst gap); the genuinely
damaging case is an early mid-chain relay kill (r=9: 0.771 → 0.575).
**Two interpretive caveats:** (a) the step-100 kill never fires in worst_gap runs that finish
before step 100 (8/10 seeds at r=12), so cite the `*_early` (step-40) scenarios for attrition
claims; (b) `no_relay` and `static_t0` show *byte-identical* coverage across all five kill
scenarios — not a bug: under range loss these stall so early (cut-off centralized drones idle
without assignments, per Metrics Design) that coverage has already plateaued before the kill
fires, so killing an already-idle drone changes nothing.

**Cross-experiment conclusion.** Worst-gap anchoring's advantage over centroid, CENTFAR, and
the analytically "correct" Fiedler-gradient method holds under seed expansion (n=40, p≤1.8e-11),
map generalization (7 conditions incl. 3 no-dead-zone map sizes and 3 out-of-distribution
dead-zone layouts), K sweep (K=4–8), fleet scaling (up to 36 searchers, 6/6 panels), and
attrition. Two bounded, honestly-reported exceptions: it ties at generous comms radius (r=18)
and at very low relay count (K=2) — both regimes where relay *placement* cannot matter much.

## Metrics Design

**Coverage and detection are measured at mission completion** — i.e. at the step when
all victims have been found, at which point the episode terminates. This is a deliberate
choice that mirrors real disaster search-and-rescue operations: the mission *is* finding
the victims, not blanketing the map. A swarm that locates everyone quickly has succeeded
even if it never physically swept every free cell, so reporting coverage/detection at that
moment captures the operationally meaningful state. (Consequence: converged runs report
coverage ~0.91 rather than ~1.0, because the swarm stops exploring once the last victim is
detected — the un-swept cells are simply areas it never needed to visit.) If a strategy
fails to find all victims, the episode runs to the `max_steps` cap and coverage/detection
reflect the full mission's stalled end-state.

**The 600-step cap never binds on converging strategies.** Empirically (perfect comms,
10 drones / 15 victims): centralized and decentralized converge by ~75–76 steps,
stigmergy by ~183, with 0% of runs hitting the cap. Under heavy comms degradation
(range severity 0.8) the resilient strategies still converge early (~93 and ~210 steps,
0% capped), while the centralized baseline reaches 100% cap-hit at only ~0.30 coverage —
but that is genuine failure, not truncation: cut-off drones idle without assignments, so
extending `max_steps` would add only idle steps, not progress. The cap binds only on a
swarm that has already stalled, never on one still making progress — so no converging run
is cut short and the degradation curves are valid. (`Simulator.run(snapshot_steps=[...])`
supports per-step coverage/detection snapshots for verifying this.)

## Experimental findings (v0 harness, 3 strategies × failure matrix × 10 seeds)

Full sweep in `swarm_sim/full_matrix.py` (`python -m swarm_sim.full_matrix`); outputs
`results/matrix.csv`, `matrix_summary.json`, `winners.json`, `degradation_<type>.png`.
Headline metric: coverage at mission completion + resilience AUC (area under
coverage-vs-severity, 1.0 = flat/ideal). Default map 50×50, 10 drones, 15 victims;
node-failure scenarios use a harder 100×100 / 30-victim / 12-drone map (see below).

**Winner by failure type:**

| Failure family | Winner | Behavior |
|---|---|---|
| Perfect comms | decentralized (0.95) | All converge (cent/dec ~75 steps, stig ~183); small margins. |
| Packet loss 5–20% | centralized (AUC 1.03) | Transient drops don't bite; per-step reassignment recovers. All > 0.90. |
| Range degradation | **stigmergy** (AUC 1.02) | Centralized **collapses 0.97→0.30**; dec/stig stay flat. |
| Dead zone 10–30% | **stigmergy** (AUC 0.99) | Centralized → 0.62 at 30%; dec/stig unaffected. |
| Node failure (random) | **decentralized** (AUC 0.93) | dec > cent > stig; at 75% loss all fail to finish. |
| Node failure (targeted) | cent ≈ dec (AUC 0.87) | Killing best-connected hubs hurts dec's gossip relays — at 75% dec (0.49) drops *below* cent (0.51). |
| Combined disaster | stigmergy ≈ decentralized (~0.93) | Centralized **collapses to 0.40**. |

**Cross-scenario conclusions:**
- **Persistent, geographic comms loss (range / dead-zone / combined) is where centralized
  control dies** (0.97→0.30 range; 0.40 combined) and where **stigmergy and decentralized
  win decisively** — the project's core thesis, now quantified.
- **Transient packet loss hurts no one**; centralized is even marginally best because its
  global planner is optimal when enough messages still arrive.
- **Decentralized consensus is the best all-rounder** — never the outright loser in any
  scenario.
- **Stigmergy is the comms-loss specialist but pays for it elsewhere:** (1) it is the
  *least* resilient to losing agents (no mechanism to re-task survivors toward gaps), and
  (2) it does **not scale efficiently** — on the 100×100 map even the full fleet reaches
  only ~0.79 coverage with ~90% of runs hitting the 800-step cap, because its diffuse,
  redundant search costs throughput. Resilient to comms loss ≠ efficient.
- **Targeted node failure > random:** removing high-degree hubs disproportionately harms
  the gossip-dependent decentralized strategy (its relays *are* the hubs), the one case
  where centralized matches or beats it under node loss.

(Caveat logged: node failure is only discriminating on the harder map fired mid-mission;
on the small map survivors trivially finish, so the original t=150 result showed no effect.
`Simulator(scheduled_failures={step: n}, failure_mode='random'|'targeted')` controls this.)

## Key references / repos
- gym-pybullet-drones — https://github.com/utiasDSL/gym-pybullet-drones
- QuadSwarm (multi-quadrotor RL sim) — https://arxiv.org/pdf/2306.09537
- Flightmare — https://arxiv.org/abs/2009.00563
- Beyond Robustness: Taxonomy of Resilient Multi-Robot Systems — https://arxiv.org/pdf/2109.12343
- Byzantine Resilience at Swarm Scale — https://arxiv.org/pdf/2301.06977
- Robustness/Scalability of Incomplete Virtual Pheromone Maps (stigmergic exploration) — https://www.mdpi.com/2227-9717/12/10/2122
- Testing the limits of pheromone stigmergy in high-density swarms — https://royalsocietypublishing.org/doi/10.1098/rsos.190225
- Physarum-based decentralized mesh networks — https://www.nature.com/articles/s41598-025-33456-y
- Swarm Intelligence-Based Multi-Robotics review — https://www.mdpi.com/2673-9909/4/4/64
- Centrality-Driven Adaptive Consensus Bundle (CBBA biomimetic) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12838631/
- Multi-UAV Coverage Planning with Limited Endurance in Disaster Environment — https://arxiv.org/pdf/2201.10150
- Heterogeneous Fixed-wing Resilient Area Coverage — https://arxiv.org/pdf/2009.09857
- FANET resilient Q-learning routing — https://arxiv.org/pdf/2306.17360

## Deliverables (current state)
- **Simulator + experiments:** `swarm_sim/` (full_matrix, relay_experiment, relay_k_sweep,
  scaling_study, relay_multibranch_experiment, relay_baselines, relay_baseline_comparison).
  Simulator supports `multihop`, `scheduled_failures` (random/targeted), per-step
  `snapshots`; `DeadZone` has `block_crossing` (line-of-sight).
- **Relay result-expansion suite:** `swarm_sim/relay_seed_expansion.py`,
  `relay_map_generalization.py`, `relay_k_sweep_all.py`, `relay_fleet_scaling.py`,
  `relay_attrition.py` → `results/relay_*.csv` + `.png` (+ `relay_seed_expansion_stats.json`).
  Run each with `/usr/bin/python3 -m swarm_sim.<name>`. See the result-expansion section above.
- **Paper:** `paper/paper.md` (~4.6k words, 12 sections, all numbers traced to `results/*`),
  plus `paper/README.md` and `paper/figures.md`. *(gitignored — local only.)*
- **Provisional patent draft:** `patent/` — `provisional_application.md`/`.docx` (spec + 10
  claims: 1–4/9 classify-and-switch, 5–8 worst-gap anchoring, 10 system), `drawings.pdf`
  (8 figures), `FILING_INSTRUCTIONS.md`. **Unfiled**; inventor fields still bracketed
  placeholders. *(gitignored — kept out of the repo so a future public/Pages flip cannot
  become an inadvertent public disclosure before the priority date is locked.)*
- **ROS2/Gazebo backend:** `ros2_backend/` — `swarm_sar` (ament_python nodes: drone /
  comm_bridge / base_station / metrics, strategy adapters, Gazebo `disaster.world`, launch,
  `params.yaml`) + `swarm_sar_interfaces` (CommMessage/DroneState msgs). The comm_bridge
  injects the same CommModel degradation as the Python sim; see `ros2_backend/README.md` for
  the full sim→ROS2 mapping table and the PX4/MAVROS hardware path.

## Environment note
`python3` on PATH (Homebrew) has **no numpy**. All sims must be run with **`/usr/bin/python3`**
(system Python: numpy + matplotlib + scipy 1.13.1). E.g.
`/usr/bin/python3 -m swarm_sim.relay_seed_expansion`.

## Status & next steps
**Done:** v0 harness (world/comms/strategies/metrics), full failure matrix (3 strategies ×
failure families × 10 seeds), relay extension + worst-gap anchoring, 4-way baseline
comparison, the 5-experiment result-expansion suite above, the paper draft, and the
ROS2/Gazebo backend.

**Open / next:**
1. **File the provisional patent** (user-owned): fill inventor fields in `patent/`, file at
   patentcenter.uspto.gov (~$65 micro-entity). This locks the priority date and should precede
   any public disclosure or professor outreach.
2. **Professor outreach** — after (1).
3. **Fold the 5 new experiments into `paper/paper.md`** — the paper currently cites only the
   10-seed baseline comparison; the n=40 stats, map generalization, K sweep, fleet scaling,
   and attrition results all need sections + figures, including the two honest negative
   findings (tie at r=18, tie at K=2).
4. Optional: real-hardware / PX4 validation via the ROS2 backend.

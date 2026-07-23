"""Baseline relay-repositioning strategies from the connectivity-maintenance literature,
implemented in this simulator so they can be compared head-to-head against the
single-farthest-agent ("worst-gap") anchoring in `relay.py` under identical conditions
(same map, seeds, failure model, K, and multihop routing).

Three baselines, matching the closest prior art found for the worst-gap mechanism:

- `CentroidRelayStrategy` — anchor the relay chain on the centroid (mean position) of all
  living searchers. The naive approach this project's own findings argue against (a central
  dead zone swallows the centroid).
- `CentroidOfFarRelayStrategy` — CENTFAR-style: anchor on the centroid of the *set* of
  searchers currently beyond nominal comm range of base ("far"/disconnected agents),
  falling back to the full-fleet centroid if none are currently far. This is a
  set-based generalization of worst-gap anchoring (center of *multiple* far nodes,
  rather than the single farthest node).
- `FiedlerRelayStrategy` — algebraic-connectivity (Fiedler value / second-smallest
  Laplacian eigenvalue) maximization, the mainstream approach in the robotics
  connectivity-maintenance literature (Cortes/Schwager/Kumar/Ayanian-style). Implemented
  as analytical gradient ascent on the Fiedler value using the Fiedler eigenvector (the
  standard control law in that literature, e.g. Kim & Mesbahi 2006), moving all relays
  simultaneously each step from one shared eigendecomposition of a soft (Gaussian-weighted)
  connectivity graph over {base, living searchers, living relays}. An earlier discretized
  greedy per-relay hill-climbing version was tried first and discarded: it reliably got
  stuck at a local optimum (verified: no single relay's unilateral one-step move improved
  the joint Fiedler value, even though a joint move of all relays clearly would), which
  would have made for an unfair strawman baseline.
"""

from __future__ import annotations

import numpy as np

from .strategies import CoordinationStrategy


def _reserve_relay_ids(drones, n_relays):
    ids = sorted(d.id for d in drones)
    return set(ids[-n_relays:]) if n_relays > 0 else set()


def _place_along_segment(base_pos, anchor_pos, n_relays):
    """n_relays points evenly spaced at fractions 1/(K+1)..K/(K+1) of base->anchor."""
    base_pos = np.asarray(base_pos, dtype=float)
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    out = []
    for i in range(n_relays):
        frac = (i + 1) / (n_relays + 1)
        out.append(tuple(base_pos + frac * (anchor_pos - base_pos)))
    return out


class CentroidRelayStrategy(CoordinationStrategy):
    """Naive baseline: anchor the relay chain on the centroid of all living searchers."""

    def __init__(self, base_strategy: CoordinationStrategy, n_relays: int = 3):
        self.base = base_strategy
        self.n_relays = n_relays
        self.name = f"centroid_relay({base_strategy.name},k={n_relays})"
        self._relay_ids = None

    def reset(self, world, drones, base):
        self._relay_ids = _reserve_relay_ids(drones, self.n_relays)
        self.base.reset(world, drones, base)

    def assign(self, world, drones, base, send, step):
        searchers = [d for d in drones if d.id not in self._relay_ids]
        relays = [d for d in drones if d.id in self._relay_ids and d.alive]
        self.base.assign(world, searchers, base, send, step)

        living = [d for d in searchers if d.alive]
        if not relays:
            return
        if not living:
            for r in relays:
                r.target = tuple(base.pos)
            return

        anchor = np.mean([d.pos for d in living], axis=0)
        for r, p in zip(relays, _place_along_segment(base.pos, anchor, len(relays))):
            r.target = p


class CentroidOfFarRelayStrategy(CoordinationStrategy):
    """CENTFAR-style baseline: anchor on the centroid of searchers currently beyond
    nominal comm range of base, falling back to the full-fleet centroid if none qualify."""

    def __init__(self, base_strategy: CoordinationStrategy, n_relays: int = 3):
        self.base = base_strategy
        self.n_relays = n_relays
        self.name = f"centfar_relay({base_strategy.name},k={n_relays})"
        self._relay_ids = None

    def reset(self, world, drones, base):
        self._relay_ids = _reserve_relay_ids(drones, self.n_relays)
        self.base.reset(world, drones, base)

    def assign(self, world, drones, base, send, step):
        searchers = [d for d in drones if d.id not in self._relay_ids]
        relays = [d for d in drones if d.id in self._relay_ids and d.alive]
        self.base.assign(world, searchers, base, send, step)

        living = [d for d in searchers if d.alive]
        if not relays:
            return
        if not living:
            for r in relays:
                r.target = tuple(base.pos)
            return

        pos = np.array([d.pos for d in living])
        dists = np.linalg.norm(pos - np.asarray(base.pos, dtype=float), axis=1)
        far_mask = dists > world.cfg.comm_radius
        anchor = pos[far_mask].mean(axis=0) if far_mask.any() else pos.mean(axis=0)
        for r, p in zip(relays, _place_along_segment(base.pos, anchor, len(relays))):
            r.target = p


# ---- Fiedler-value (algebraic connectivity) maximization -------------------

def _fiedler_value_and_vector(positions, sigma):
    """Second-smallest eigenpair of the Gaussian-weighted-complete-graph Laplacian."""
    pts = np.asarray(positions, dtype=float)
    n = len(pts)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2.0 * sigma ** 2))
    np.fill_diagonal(w, 0.0)
    lap = np.diag(w.sum(axis=1)) - w
    eigvals, eigvecs = np.linalg.eigh(lap)
    return eigvals[1], eigvecs[:, 1], w


def _repulsion(pts, idx, min_dist):
    """Standard inter-agent repulsion (active only within min_dist), used alongside the
    Fiedler-ascent gradient. Without it, agents driven by an identical shared gradient
    (as all relays are here) provably collapse onto the same point whenever they start
    near each other — a well-known degeneracy of pure multi-agent gradient controllers,
    which the connectivity-maintenance literature addresses by combining the connectivity
    gradient with a collision/redundancy-avoiding repulsive potential."""
    xi = pts[idx]
    force = np.zeros(2)
    for j in range(len(pts)):
        if j == idx:
            continue
        delta = xi - pts[j]
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            # exact tie (e.g. two relays spawned on the same cell): deterministic
            # tie-break direction from the pair's indices, so it's reproducible.
            angle = idx * 2.399963229728653 + j * 0.6180339887498949
            force += np.array([np.cos(angle), np.sin(angle)])
        elif dist < min_dist:
            force += (delta / dist) * (min_dist - dist) / min_dist
    return force


class FiedlerRelayStrategy(CoordinationStrategy):
    """Mainstream baseline: reposition relays each step via analytical gradient ascent
    on the network's algebraic connectivity (Fiedler value), following the standard
    connectivity-maintenance control law used in the literature (e.g. Kim & Mesbahi 2006;
    Cortes-style CBF/potential formulations): each node's velocity is driven by
    d(lambda_2)/dx_i = sum_j w_ij * (v_i - v_j)^2 * (x_j - x_i), where v is the Fiedler
    eigenvector — i.e. each relay is pulled toward nodes on the "other side" of the
    graph's weakest partition, weighted by current connection strength. Combined with a
    standard inter-relay repulsion term (see `_repulsion`), since relays sharing an
    identical gradient law otherwise collapse onto the same point (verified empirically:
    without repulsion, all K relays converge to one identical position, wasting K-1 of
    them). This two-term (attraction + repulsion) combination is the standard form used
    in the literature's potential-field connectivity controllers, not an ad hoc addition."""

    def __init__(self, base_strategy: CoordinationStrategy, n_relays: int = 3,
                 repulsion_dist: float | None = None, repulsion_strength: float = 1.5):
        self.base = base_strategy
        self.n_relays = n_relays
        self.repulsion_dist = repulsion_dist
        self.repulsion_strength = repulsion_strength
        self.name = f"fiedler_relay({base_strategy.name},k={n_relays})"
        self._relay_ids = None

    def reset(self, world, drones, base):
        self._relay_ids = _reserve_relay_ids(drones, self.n_relays)
        self.base.reset(world, drones, base)

    def assign(self, world, drones, base, send, step):
        searchers = [d for d in drones if d.id not in self._relay_ids]
        relays = [d for d in drones if d.id in self._relay_ids and d.alive]
        self.base.assign(world, searchers, base, send, step)
        if not relays:
            return

        living_searchers = [d for d in searchers if d.alive]
        sigma = max(1.0, world.cfg.comm_radius)
        rep_dist = self.repulsion_dist if self.repulsion_dist is not None else sigma / 2.0
        nodes = [np.asarray(base.pos, dtype=float)] + \
                [np.array(d.pos, dtype=float) for d in living_searchers] + \
                [np.array(r.pos, dtype=float) for r in relays]
        n_fixed = 1 + len(living_searchers)
        pts = np.stack(nodes)

        _, v, w = _fiedler_value_and_vector(pts, sigma)

        for k, r in enumerate(relays):
            i = n_fixed + k
            diff = pts - pts[i]                      # (n,2) x_j - x_i
            coeff = w[i] * (v[i] - v) ** 2            # (n,)
            grad = (coeff[:, None] * diff).sum(axis=0)
            grad = grad + self.repulsion_strength * _repulsion(pts, i, rep_dist)
            norm = np.linalg.norm(grad)
            if norm < 1e-9:
                continue
            step_dir = grad / norm
            r.target = tuple(pts[i] + r.cfg.speed * step_dir)

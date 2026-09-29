"""Pair schedulers for pairwise judging.

Three flavours:

`build_schedule`
    Uniformly random d-regular graph (n*d/2 pairs, every uid in exactly
    d pairs).

`build_extension_schedule`
    Given an existing d-regular run, return a k-regular graph
    edge-disjoint from it. Union is (d+k)-regular. Lets a saturating
    run grow without invalidating cached calls.

`build_soft_regular_schedule`
    Streaming weighted scheduler: pick each (a, b) with vertex weight
    ∝ 1/(1+N[u])^x (N = current degree). The x knob interpolates between
    iid-with-replacement (x=0) and degree-regular (x→∞). No exact-degree
    guarantee, but naturally extensible — same seed + larger budget
    produces the same prefix. The schedule used to collect the caches.

All schedulers are deterministic given `(uids, …, master_seed,
namespace)`. The namespace lets two pilots draw independent schedules
from the same uid set without colliding seeds; extensions seed off
`f"{namespace}|ext{N}"`.
"""

from __future__ import annotations

import random
from typing import Sequence

import networkx as nx

from core.seeds import seed_from_string


def build_schedule(
    uids: Sequence[str],
    *,
    degree: int = 10,
    master_seed: int = 0,
    namespace: str = "pair_schedule",
) -> list[tuple[str, str]]:
    items = list(uids)
    n = len(items)
    if degree >= n:
        raise ValueError(f"degree={degree} must be < n={n}")
    if (degree * n) % 2:
        raise ValueError(
            f"degree*n must be even (degree={degree}, n={n}); "
            f"pick a different degree"
        )

    seed_int = seed_from_string(master_seed, namespace)
    G = nx.random_regular_graph(degree, n, seed=seed_int)

    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for u, v in G.edges():
        a, b = items[u], items[v]
        key = frozenset((a, b))
        if key in seen:
            raise RuntimeError(f"duplicate edge {key} in random regular graph")
        seen.add(key)
        pairs.append((a, b))

    expected = n * degree // 2
    if len(pairs) != expected:
        raise RuntimeError(
            f"got {len(pairs)} pairs, expected {expected} for "
            f"degree={degree}, n={n}"
        )
    return pairs


def build_extension_schedule(
    existing_pairs: Sequence[tuple[str, str]],
    uids: Sequence[str],
    *,
    degree: int,
    master_seed: int = 0,
    namespace: str,
    max_attempts: int = 200,
) -> list[tuple[str, str]]:
    """Return a `degree`-regular graph on `uids`, edge-disjoint from
    `existing_pairs` (each uid gets exactly `degree` *new* comparisons).
    The union with `existing_pairs` is (prior_degree + degree)-regular;
    the prior degree is not re-checked (caller's responsibility).

    Built by sequential Steger-Wormald-style pairing on the complement,
    restarting with the next seed on a dead end (see the loop below).

    Statistical note: the union graph is drawn from the slice containing
    the prior schedule as a subgraph, not uniformly from all such regular
    graphs. Irrelevant for pair-level statistics (each pair is an
    independent observation); matters only if you analyse the schedule's
    own joint structure.
    """
    items = list(uids)
    n = len(items)
    if degree <= 0:
        raise ValueError(f"extension degree must be positive (got {degree})")
    if degree >= n:
        raise ValueError(f"degree={degree} must be < n={n}")
    if (degree * n) % 2:
        raise ValueError(
            f"degree*n must be even (degree={degree}, n={n}); "
            f"pick a different extension degree"
        )

    forbidden: set[frozenset[str]] = {frozenset(e) for e in existing_pairs}

    existing_deg: dict[str, int] = {u: 0 for u in items}
    for a, b in existing_pairs:
        if a in existing_deg:
            existing_deg[a] += 1
        if b in existing_deg:
            existing_deg[b] += 1
    if existing_deg:
        min_complement = min(n - 1 - existing_deg[u] for u in items)
        if min_complement < degree:
            tight = next(
                u for u in items
                if n - 1 - existing_deg[u] == min_complement
            )
            raise ValueError(
                f"extension impossible: uid {tight!r} has only "
                f"{min_complement} available partners in the complement, "
                f"need {degree}. Reduce the extension degree or accept "
                f"that the existing schedule is nearly complete."
            )

    base_seed = seed_from_string(master_seed, namespace)

    # At each step pair the vertex with the most remaining stubs to a random
    # valid partner. (Configuration-model shuffle-then-pair rejects nearly
    # always at n=200, k=10 — too many multi-edge collisions in 2000 stubs.)
    for attempt in range(max_attempts):
        rng = random.Random((base_seed ^ attempt) & 0xFFFFFFFFFFFFFFFF)
        remaining: dict[str, int] = {u: degree for u in items}
        edges: set[frozenset[str]] = set()
        stuck = False
        while True:
            available = [u for u in items if remaining[u] > 0]
            if not available:
                break
            max_rem = max(remaining[u] for u in available)
            top = [u for u in available if remaining[u] == max_rem]
            a = rng.choice(top)
            candidates = [
                u for u in available
                if u != a
                and remaining[u] > 0
                and frozenset((a, u)) not in forbidden
                and frozenset((a, u)) not in edges
            ]
            if not candidates:
                stuck = True
                break
            b = rng.choice(candidates)
            edges.add(frozenset((a, b)))
            remaining[a] -= 1
            remaining[b] -= 1
        if not stuck and sum(remaining.values()) == 0:
            result: list[tuple[str, str]] = []
            for key in edges:
                a, b = sorted(key)
                result.append((a, b))
            result.sort()
            return result

    raise RuntimeError(
        f"build_extension_schedule failed after {max_attempts} attempts. "
        f"Parameters: degree={degree}, n={n}, "
        f"existing_edges={len(existing_pairs)}. The complement graph is "
        f"likely too constrained; reduce the extension degree or start "
        f"fresh with a different master_seed."
    )


def build_soft_regular_schedule(
    uids: Sequence[str],
    *,
    budget: int,
    x: float = 2.0,
    master_seed: int = 0,
    namespace: str = "soft_pair_schedule",
) -> list[tuple[str, str]]:
    """Streaming weighted pair scheduler. Returns `budget` unique unordered
    pairs, each drawn sequentially with vertex weight ∝ 1/(1 + N[u])^x
    (N[u] = pairs u already appears in). The `x` knob interpolates from
    iid-with-replacement (x=0) to degree-regular (x→∞). Self-loops and
    duplicate pairs are excluded (simple graph).

    Deterministic given `(master_seed, namespace, budget, x)`, and
    naturally extensible: same seed + larger budget reproduces the prefix.

    Args:
        uids: vertex set.
        budget: pairs to draw; 0 <= budget <= n*(n-1)/2.
        x: uniformity exponent, x >= 0.
        master_seed, namespace: deterministic seed inputs.

    Raises ValueError if budget exceeds the simple-graph capacity or x < 0.
    """
    items = list(uids)
    n = len(items)
    max_pairs = n * (n - 1) // 2
    if budget < 0:
        raise ValueError(f"budget must be non-negative (got {budget})")
    if budget > max_pairs:
        raise ValueError(
            f"budget {budget} exceeds simple-graph capacity "
            f"{max_pairs} for n={n}"
        )
    if x < 0:
        raise ValueError(f"x must be non-negative (got {x})")

    import math as _math
    seed_int = seed_from_string(master_seed, namespace)
    rng = random.Random(seed_int)

    N: dict[str, int] = {u: 0 for u in items}
    partners: dict[str, set[str]] = {u: set() for u in items}
    pairs: list[tuple[str, str]] = []

    def _weights(candidates: list[str]) -> list[float]:
        """Numerically stable for any x: exp(-x*log(1+N) - max). Bottom
        entries underflowing to 0 is correct (near-zero prob at large x)."""
        if x == 0:
            return [1.0] * len(candidates)
        log_w = [-x * _math.log(1.0 + N[u]) for u in candidates]
        m = max(log_w)
        return [_math.exp(lw - m) for lw in log_w]

    while len(pairs) < budget:
        a_candidates = [u for u in items if len(partners[u]) < n - 1]
        if not a_candidates:
            raise RuntimeError(
                f"soft scheduler exhausted at {len(pairs)}/{budget} "
                f"pairs (every uid is paired with every other). "
                f"Reduce budget or increase n."
            )
        a = _weighted_choice(rng, a_candidates, _weights(a_candidates))

        avail_b = [
            u for u in items
            if u != a and u not in partners[a]
        ]
        if not avail_b:
            # Should not happen given a_candidates filter, but defensive.
            continue
        b = _weighted_choice(rng, avail_b, _weights(avail_b))

        a_out, b_out = (a, b) if a <= b else (b, a)
        pairs.append((a_out, b_out))
        partners[a].add(b)
        partners[b].add(a)
        N[a] += 1
        N[b] += 1

    return pairs


def _weighted_choice(rng: random.Random, items: list[str], weights: list[float]) -> str:
    """Pick one item with probability proportional to weights. Falls
    back to uniform if every weight is zero (e.g. all-uids-saturated
    edge case at x→∞ when only a degenerate set remains)."""
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    cum = 0.0
    for item, w in zip(items, weights):
        cum += w
        if r < cum:
            return item
    return items[-1]

# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FLIPS cluster-aware participant selector.

Implements round-robin / fairness-aware client selection across clusters as
described in the FLIPS paper.  The selector is intentionally pure-Python with
no Flower API dependency so it can be unit-tested in isolation.

Selection algorithm
-------------------
1. Distribute ``clients_per_round`` slots evenly across the known clusters
   (base allocation = slots // n_clusters; remainder slots go to clusters
   with the most clients, deterministically).
2. Within each cluster, prefer clients with the lowest ``pick_count``; ties
   are broken by client_id lexicographic order (deterministic under sorting).
3. After filling cluster quotas, any remaining slots (because some clusters
   have fewer available clients than their quota) are filled from the global
   pool, again preferring the least-picked clients.
4. The final list is ordered by cluster_id for reproducibility.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from flwr.server.flips.server.metadata_registry import ClientMetadata, MetadataRegistry

SelectionResult = Tuple[List[str], Dict[int, List[str]]]
"""``(selected_ids, per_cluster_selected)``"""


class FlipsSelector:
    """Select clients for a training round using FLIPS cluster-aware fairness.

    Parameters
    ----------
    registry:
        Live :class:`MetadataRegistry` instance shared with the strategy.
    seed:
        Optional integer seed used to break all ties deterministically.
        When ``None``, Python's built-in sort (stable, deterministic for
        equal-key items) is used directly.
    """

    def __init__(
        self,
        registry: MetadataRegistry,
        seed: Optional[int] = None,
    ) -> None:
        self._registry = registry
        self._seed = seed

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def select(
        self,
        clients_per_round: int,
        available_ids: Optional[FrozenSet[str]] = None,
        excluded_ids: Optional[Set[str]] = None,
    ) -> SelectionResult:
        """Return a list of client IDs selected for the next round.

        Parameters
        ----------
        clients_per_round:
            Target number of clients to select.
        available_ids:
            If provided, selection is limited to this set of IDs.  When
            ``None``, all clients in the registry are considered available.
        excluded_ids:
            Client IDs that must not be selected (e.g., known stragglers
            that have already been excluded from overprovisioning).

        Returns
        -------
        SelectionResult
            A tuple of ``(selected_ids, per_cluster_map)`` where
            ``per_cluster_map`` maps ``cluster_id -> [client_id, ...]``.
        """
        excluded = excluded_ids or set()

        all_meta: Dict[str, ClientMetadata] = self._registry.all()
        if available_ids is not None:
            all_meta = {k: v for k, v in all_meta.items() if k in available_ids}
        # Remove excluded
        all_meta = {k: v for k, v in all_meta.items() if k not in excluded}

        if clients_per_round <= 0 or not all_meta:
            return [], {}

        # Cluster the available clients
        cluster_buckets: Dict[int, List[str]] = {}
        unclustered: List[str] = []
        for cid, meta in all_meta.items():
            if meta.cluster_id == -1:
                unclustered.append(cid)
            else:
                cluster_buckets.setdefault(meta.cluster_id, []).append(cid)

        # Treat unclustered clients as a virtual cluster (-1)
        if unclustered:
            cluster_buckets[-1] = unclustered

        cluster_ids = sorted(cluster_buckets.keys())
        n_clusters = len(cluster_ids)
        n_available = sum(len(v) for v in cluster_buckets.values())
        target = min(clients_per_round, n_available)

        if n_clusters == 0:
            return [], {}

        # ---------------------------------------------------------------- #
        # Step 1: compute per-cluster quotas                                 #
        # ---------------------------------------------------------------- #
        quotas = self._compute_quotas(cluster_ids, cluster_buckets, target)

        # ---------------------------------------------------------------- #
        # Step 2: select least-picked clients within each cluster quota      #
        # ---------------------------------------------------------------- #
        selected: List[str] = []
        per_cluster_selected: Dict[int, List[str]] = {}
        overflow_pool: List[str] = []

        for cid_cluster in cluster_ids:
            candidates = sorted(
                cluster_buckets[cid_cluster],
                key=lambda c: (all_meta[c].pick_count, c),
            )
            quota = quotas.get(cid_cluster, 0)
            chosen = candidates[:quota]
            leftover = candidates[quota:]
            selected.extend(chosen)
            per_cluster_selected[cid_cluster] = list(chosen)
            overflow_pool.extend(leftover)

        # ---------------------------------------------------------------- #
        # Step 3: fill remaining slots from the overflow pool               #
        # ---------------------------------------------------------------- #
        remaining = target - len(selected)
        if remaining > 0 and overflow_pool:
            overflow_pool.sort(key=lambda c: (all_meta[c].pick_count, c))
            extras = overflow_pool[:remaining]
            selected.extend(extras)
            for c in extras:
                cluster_id = all_meta[c].cluster_id
                per_cluster_selected.setdefault(cluster_id, []).append(c)

        # Sort final list by (cluster_id, pick_count, client_id) for determinism
        selected.sort(key=lambda c: (all_meta[c].cluster_id, all_meta[c].pick_count, c))
        return selected, per_cluster_selected

    def select_overprovision(
        self,
        clients_per_round: int,
        straggler_rate: float,
        affected_clusters: Optional[List[int]] = None,
        available_ids: Optional[FrozenSet[str]] = None,
    ) -> SelectionResult:
        """Select an overprovisioned cohort to compensate for expected stragglers.

        The extra slots are filled preferentially from the same cluster(s)
        where stragglers were last observed (FLIPS paper, Sec. 4.3).

        Parameters
        ----------
        clients_per_round:
            Desired number of *usable* results after straggler drop-out.
        straggler_rate:
            Fraction in [0, 1) of clients expected to straggle.
        affected_clusters:
            Cluster IDs from which extra clients should be drawn first.
            If ``None``, extra clients come from the global pool.
        available_ids:
            Optional restriction of eligible client IDs.

        Returns
        -------
        SelectionResult
            The enlarged cohort selection.
        """
        extra = math.ceil(clients_per_round * straggler_rate)
        total = clients_per_round + extra

        if not affected_clusters:
            return self.select(total, available_ids=available_ids)

        # Base selection
        base_selected, per_cluster = self.select(
            clients_per_round, available_ids=available_ids
        )
        base_set = set(base_selected)
        remaining_slots = total - len(base_selected)

        if remaining_slots <= 0:
            return base_selected, per_cluster

        # Fill extra slots from the affected clusters first
        all_meta = self._registry.all()
        extra_candidates: List[str] = []
        for cluster_id in affected_clusters:
            for cid, meta in all_meta.items():
                if (
                    meta.cluster_id == cluster_id
                    and cid not in base_set
                    and (available_ids is None or cid in available_ids)
                ):
                    extra_candidates.append(cid)

        extra_candidates.sort(key=lambda c: (all_meta[c].pick_count, c))
        extras = extra_candidates[:remaining_slots]
        remaining_slots -= len(extras)

        # Further fill from global pool if needed
        if remaining_slots > 0:
            global_pool = [
                cid
                for cid, meta in all_meta.items()
                if cid not in base_set
                and cid not in set(extras)
                and (available_ids is None or cid in available_ids)
            ]
            global_pool.sort(key=lambda c: (all_meta[c].pick_count, c))
            extras.extend(global_pool[:remaining_slots])

        combined = base_selected + extras
        combined_set = set(combined)
        for c in extras:
            cluster_id = all_meta[c].cluster_id
            per_cluster.setdefault(cluster_id, []).append(c)

        combined.sort(key=lambda c: (all_meta[c].cluster_id, all_meta[c].pick_count, c))
        return combined, per_cluster

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #


    @staticmethod
    def _compute_quotas(
        cluster_ids: List[int],
        cluster_buckets: Dict[int, List[str]],
        target: int,
    ) -> Dict[int, int]:
        """Distribute ``target`` slots across clusters fairly.

        Base allocation is ``target // n_clusters``; remainder slots are
        given, one each, to the clusters with the most available clients
        (deterministic tiebreak: smaller cluster_id wins).
        """
        n = len(cluster_ids)
        base = target // n
        remainder = target % n

        # Sort by available clients descending, then cluster_id ascending for tiebreak
        sorted_clusters = sorted(
            cluster_ids,
            key=lambda c: (-len(cluster_buckets[c]), c),
        )

        quotas: Dict[int, int] = {}
        for i, cid in enumerate(cluster_ids):
            quotas[cid] = base

        for i in range(remainder):
            quotas[sorted_clusters[i]] += 1

        # Cap quotas at available clients per cluster
        for cid in cluster_ids:
            quotas[cid] = min(quotas[cid], len(cluster_buckets[cid]))

        return quotas


# Avoid circular import when math is used in select_overprovision
import math  # noqa: E402

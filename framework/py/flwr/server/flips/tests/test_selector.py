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
"""Unit tests for FlipsSelector (server-side participant selection)."""

import pytest

from flwr.server.flips.server.metadata_registry import ClientMetadata, MetadataRegistry
from flwr.server.flips.server.selector import FlipsSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_registry(
    n_clients: int,
    n_clusters: int,
    pick_counts: dict[str, int] | None = None,
) -> MetadataRegistry:
    """Build a registry with ``n_clients`` evenly distributed across clusters."""
    registry = MetadataRegistry()
    pick_counts = pick_counts or {}

    for i in range(n_clients):
        cid = f"client_{i:03d}"
        registry.register(cid)
        cluster_id = i % n_clusters
        registry.update_cluster(cid, cluster_id)
        meta = registry.get(cid)
        assert meta is not None
        meta.pick_count = pick_counts.get(cid, 0)

    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlipsSelector:
    def test_select_returns_correct_count(self) -> None:
        """Selector returns exactly the requested number of clients."""
        registry = _build_registry(20, 4)
        selector = FlipsSelector(registry, seed=0)
        selected, _ = selector.select(8)
        assert len(selected) == 8

    def test_no_duplicate_selections(self) -> None:
        """Each client appears at most once in the selection."""
        registry = _build_registry(20, 4)
        selector = FlipsSelector(registry)
        selected, _ = selector.select(12)
        assert len(selected) == len(set(selected))

    def test_fair_cluster_coverage(self) -> None:
        """When slots == multiple of clusters, all clusters are represented."""
        n_clusters = 4
        n_clients = 20  # 5 per cluster
        registry = _build_registry(n_clients, n_clusters)
        selector = FlipsSelector(registry, seed=0)
        _, per_cluster = selector.select(8)  # 2 per cluster
        assert sorted(per_cluster.keys()) == list(range(n_clusters))
        for cluster_id in range(n_clusters):
            assert len(per_cluster[cluster_id]) == 2

    def test_lower_pick_count_preferred(self) -> None:
        """Clients with lower pick counts are selected before higher-picked peers."""
        registry = MetadataRegistry()
        for i in range(6):
            cid = f"c{i}"
            registry.register(cid)
            registry.update_cluster(cid, 0)  # single cluster
            meta = registry.get(cid)
            assert meta is not None
            meta.pick_count = i  # c0=0 through c5=5

        selector = FlipsSelector(registry, seed=0)
        selected, _ = selector.select(3)
        # Should prefer c0, c1, c2 (lowest pick counts)
        assert set(selected) == {"c0", "c1", "c2"}

    def test_selection_respects_available_ids(self) -> None:
        """Selection is restricted to the available_ids set."""
        registry = _build_registry(20, 2)
        allowed = frozenset({f"client_{i:03d}" for i in range(10)})
        selector = FlipsSelector(registry)
        selected, _ = selector.select(6, available_ids=allowed)
        assert all(c in allowed for c in selected)

    def test_excluded_ids_not_selected(self) -> None:
        """Excluded clients are never in the selection."""
        registry = _build_registry(20, 2)
        excluded = {f"client_{i:03d}" for i in range(10)}
        selector = FlipsSelector(registry)
        selected, _ = selector.select(6, excluded_ids=excluded)
        assert not set(selected) & excluded

    def test_select_fewer_than_available(self) -> None:
        """Requesting more clients than available caps at n_available."""
        registry = _build_registry(5, 2)
        selector = FlipsSelector(registry)
        selected, _ = selector.select(100)
        assert len(selected) == 5

    def test_empty_registry_returns_empty(self) -> None:
        registry = MetadataRegistry()
        selector = FlipsSelector(registry)
        selected, per_cluster = selector.select(5)
        assert selected == []
        assert per_cluster == {}

    def test_single_cluster(self) -> None:
        """Works correctly when all clients are in the same cluster."""
        registry = _build_registry(10, 1)
        selector = FlipsSelector(registry, seed=0)
        selected, per_cluster = selector.select(4)
        assert len(selected) == 4
        assert len(set(selected)) == 4

    def test_deterministic_under_seed(self) -> None:
        """Two calls with the same registry state return the same selection."""
        registry = _build_registry(30, 5)
        selector_a = FlipsSelector(registry, seed=99)
        selector_b = FlipsSelector(registry, seed=99)
        sel_a, _ = selector_a.select(10)
        sel_b, _ = selector_b.select(10)
        assert sel_a == sel_b

    def test_overprovision_selects_extra_clients(self) -> None:
        """select_overprovision returns more clients than clients_per_round."""
        registry = _build_registry(30, 3)
        selector = FlipsSelector(registry, seed=0)
        base_selected, _ = selector.select(6)
        op_selected, _ = selector.select_overprovision(6, straggler_rate=0.33)
        assert len(op_selected) > len(base_selected)

    def test_overprovision_prefers_affected_clusters(self) -> None:
        """Extra clients during overprovisioning come from affected clusters first."""
        registry = _build_registry(30, 3)
        selector = FlipsSelector(registry, seed=0)
        op_selected, per_cluster = selector.select_overprovision(
            clients_per_round=6,
            straggler_rate=0.33,
            affected_clusters=[0],
        )
        # Cluster 0 should get at least its fair share
        cluster_0_count = per_cluster.get(0, [])
        assert len(cluster_0_count) > 0

    def test_unclustered_clients_get_virtual_cluster(self) -> None:
        """Unclustered clients (cluster_id == -1) participate in selection."""
        registry = MetadataRegistry()
        for i in range(5):
            registry.register(f"uc{i}")
            # Leave cluster_id at -1 (default)

        selector = FlipsSelector(registry)
        selected, _ = selector.select(3)
        assert len(selected) == 3

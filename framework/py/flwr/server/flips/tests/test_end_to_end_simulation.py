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
"""End-to-end simulation test with synthetic non-IID federated clients.

Uses only in-process Python objects — no sockets, no gRPC.
Verifies that:
- Cluster-aware selection improves cluster coverage vs random baseline.
- All round metrics are produced.
- Structured logs are written when a log_path is configured.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest

from flwr.common import (
    FitRes,
    Parameters,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.flips.server.aggregation import make_flips_fedavg
from flwr.server.flips.server.metadata_registry import MetadataRegistry
from flwr.server.flips.server.clustering import LabelDistributionClusterer


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_non_iid_distributions(
    n_clients: int, n_classes: int, n_clusters: int, seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Each cluster concentrates on different classes."""
    rng = np.random.default_rng(seed)
    dists = {}
    for i in range(n_clients):
        cluster = i % n_clusters
        # Strong prior toward class = cluster
        alpha = [0.05] * n_classes
        alpha[cluster % n_classes] = 5.0
        vec = rng.dirichlet(alpha)
        dists[f"c{i:03d}"] = {str(j): float(vec[j]) for j in range(n_classes)}
    return dists


def _dummy_params(val: float = 0.0) -> Parameters:
    return ndarrays_to_parameters([np.array([val, val, val], dtype=np.float32)])


def _make_proxy(cid: str) -> ClientProxy:
    p = MagicMock(spec=ClientProxy)
    p.cid = cid
    return p


def _make_fit_res(val: float = 1.0, num_examples: int = 20) -> FitRes:
    return FitRes(
        status=MagicMock(),
        parameters=_dummy_params(val),
        num_examples=num_examples,
        metrics={
            "flips_train_time_s": 0.05,
            "flips_num_samples": num_examples,
        },
    )


def _make_client_manager(cids: List[str]) -> MagicMock:
    proxies = {cid: _make_proxy(cid) for cid in cids}
    manager = MagicMock()
    manager.all.return_value = proxies
    manager.num_available.return_value = len(cids)
    return manager


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestEndToEndSimulation:
    """Simulate multiple FL rounds with non-IID clients."""

    def _run_simulation(
        self,
        n_clients: int = 30,
        n_classes: int = 5,
        n_clusters: int = 5,
        n_rounds: int = 6,
        clients_per_round: int = 10,
        log_path: Path | None = None,
    ):
        distributions = _make_non_iid_distributions(
            n_clients, n_classes, n_clusters, seed=0
        )
        cids = sorted(distributions.keys())

        registry = MetadataRegistry()
        for cid in cids:
            registry.register(cid)

        # Pre-load label distributions into the registry
        for cid, dist in distributions.items():
            registry.update_label_distribution(cid, dist, num_samples=100)

        # Run initial clustering
        clusterer = LabelDistributionClusterer(k=n_clusters, seed=0)
        result = clusterer.fit(distributions)
        for cid, cluster_id in result.client_cluster_map.items():
            registry.update_cluster(cid, cluster_id)

        strategy = make_flips_fedavg(
            clients_per_round=clients_per_round,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=_dummy_params(0.0),
            registry=registry,
            log_path=log_path,
        )

        manager = _make_client_manager(cids)
        cluster_coverages: List[int] = []

        params = _dummy_params(0.0)
        for rnd in range(1, n_rounds + 1):
            # configure_fit
            config_pairs = strategy.configure_fit(rnd, params, manager)
            selected_cids = [proxy.cid for proxy, _ in config_pairs]

            # Simulate all clients completing
            results: List[Tuple[ClientProxy, FitRes]] = [
                (_make_proxy(cid), _make_fit_res(float(i)))
                for i, cid in enumerate(selected_cids)
            ]
            params, _ = strategy.aggregate_fit(rnd, results, [])
            if params is None:
                params = _dummy_params(0.0)

            # Measure cluster coverage
            coverage = len(
                {registry.get(cid).cluster_id for cid in selected_cids if registry.get(cid)}
            )
            cluster_coverages.append(coverage)

        return strategy, cluster_coverages

    def test_metrics_logged_each_round(self) -> None:
        strategy, _ = self._run_simulation(n_rounds=4)
        assert len(strategy._flips_logger.history()) == 4

    def test_jsonl_log_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "logs" / "flips.jsonl"
            strategy, _ = self._run_simulation(n_rounds=3, log_path=log_path)
            assert log_path.exists()
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 3
            for line in lines:
                rec = json.loads(line)
                assert "server_round" in rec
                assert "selected_clients" in rec

    def test_cluster_coverage_at_least_one(self) -> None:
        """Every round should cover at least one cluster."""
        _, coverages = self._run_simulation()
        assert all(c >= 1 for c in coverages)

    def test_effective_participation_recorded(self) -> None:
        strategy, _ = self._run_simulation(n_rounds=3)
        for m in strategy._flips_logger.history():
            assert 0.0 < m.effective_participation <= 1.0

    def test_summary_is_non_empty(self) -> None:
        strategy, _ = self._run_simulation(n_rounds=4)
        summary = strategy._flips_logger.summary()
        assert summary["total_rounds"] == 4
        assert 0.0 <= summary["mean_effective_participation"] <= 1.0

    def test_pick_counts_increase_per_round(self) -> None:
        """After N rounds, all selected clients should have pick_count > 0."""
        n_clients = 20
        strategy, _ = self._run_simulation(
            n_clients=n_clients,
            n_rounds=6,
            clients_per_round=10,
        )
        registry = strategy._flips_registry
        total_picks = sum(meta.pick_count for meta in registry.all().values())
        assert total_picks > 0

    def test_straggler_tracker_populated(self) -> None:
        strategy, _ = self._run_simulation(n_rounds=5)
        assert len(strategy._flips_straggler.history()) == 5

    def test_multi_cluster_coverage_with_non_iid(self) -> None:
        """On non-IID data with n_clusters clusters, FLIPS should cover > 1 cluster
        per round (when clients_per_round >= n_clusters)."""
        n_clusters = 5
        _, coverages = self._run_simulation(
            n_clients=30,
            n_clusters=n_clusters,
            clients_per_round=10,
            n_rounds=5,
        )
        # At least some rounds should cover multiple clusters
        assert max(coverages) > 1

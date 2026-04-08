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
"""Tests confirming FLIPS selection is compatible with FedAvg/FedProx/FedYogi
aggregation.

These tests use lightweight mock objects — no real network, no GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.flips.server.aggregation import (
    make_flips_fedavg,
    make_flips_fedprox,
    make_flips_fedyogi,
)
from flwr.server.flips.server.metadata_registry import MetadataRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_parameters(val: float = 0.0) -> Parameters:
    """Return a Parameters object containing a single 3-element array."""
    return ndarrays_to_parameters([np.array([val, val, val], dtype=np.float32)])


def _make_proxy(cid: str) -> ClientProxy:
    proxy = MagicMock(spec=ClientProxy)
    proxy.cid = cid
    return proxy


def _make_fit_res(val: float = 1.0, num_examples: int = 10) -> FitRes:
    return FitRes(
        status=MagicMock(code=MagicMock(value=0)),
        parameters=_dummy_parameters(val),
        num_examples=num_examples,
        metrics={"flips_train_time_s": 0.1, "flips_num_samples": num_examples},
    )


def _make_client_manager(
    cids: List[str],
    registry: Optional[MetadataRegistry] = None,
) -> MagicMock:
    """Build a mock ClientManager."""
    proxies = {cid: _make_proxy(cid) for cid in cids}
    manager = MagicMock()
    manager.all.return_value = proxies
    manager.num_available.return_value = len(cids)
    manager.sample.return_value = list(proxies.values())
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlipsAggregationCompat:
    """Verify FLIPS selection produces valid inputs for all aggregation paths."""

    def _run_one_round(self, strategy, n_clients: int = 10, n_results: int = 8):
        """Run one configure_fit + aggregate_fit cycle with mock data."""
        cids = [f"c{i}" for i in range(n_clients)]
        manager = _make_client_manager(cids)

        # Register clients by populating the FLIPS registry
        for i, cid in enumerate(cids):
            strategy._flips_registry.register(cid)
            strategy._flips_registry.update_cluster(cid, i % 2)

        params = _dummy_parameters(0.0)
        config_pairs = strategy.configure_fit(1, params, manager)
        assert len(config_pairs) > 0

        # Simulate some results
        results: List[Tuple[ClientProxy, FitRes]] = [
            (_make_proxy(cids[i]), _make_fit_res(float(i)))
            for i in range(n_results)
        ]
        failures = []

        agg_params, metrics = strategy.aggregate_fit(1, results, failures)
        return agg_params, metrics

    def test_fedavg_compat(self) -> None:
        strategy = make_flips_fedavg(
            clients_per_round=6,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=_dummy_parameters(0.0),
        )
        agg_params, _ = self._run_one_round(strategy)
        assert agg_params is not None
        arr = parameters_to_ndarrays(agg_params)[0]
        assert arr.shape == (3,)

    def test_fedprox_compat(self) -> None:
        strategy = make_flips_fedprox(
            clients_per_round=6,
            proximal_mu=0.01,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=_dummy_parameters(0.0),
        )
        agg_params, _ = self._run_one_round(strategy)
        assert agg_params is not None

    def test_fedyogi_compat(self) -> None:
        initial = _dummy_parameters(0.0)
        strategy = make_flips_fedyogi(
            clients_per_round=6,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=initial,
        )
        agg_params, _ = self._run_one_round(strategy)
        assert agg_params is not None

    def test_configure_fit_returns_only_selected_clients(self) -> None:
        """configure_fit must not return more pairs than selected clients."""
        strategy = make_flips_fedavg(
            clients_per_round=5,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=_dummy_parameters(),
        )
        cids = [f"c{i}" for i in range(20)]
        manager = _make_client_manager(cids)
        for i, cid in enumerate(cids):
            strategy._flips_registry.register(cid)
            strategy._flips_registry.update_cluster(cid, i % 3)

        params = _dummy_parameters()
        config_pairs = strategy.configure_fit(1, params, manager)
        assert len(config_pairs) <= 5

    def test_label_dist_injected_into_config_round1(self) -> None:
        """Round 1 FitIns config should carry the label-distribution request flag."""
        strategy = make_flips_fedavg(
            clients_per_round=4,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=_dummy_parameters(),
        )
        cids = [f"c{i}" for i in range(10)]
        manager = _make_client_manager(cids)
        for i, cid in enumerate(cids):
            strategy._flips_registry.register(cid)

        config_pairs = strategy.configure_fit(1, _dummy_parameters(), manager)
        for _, fit_ins in config_pairs:
            assert fit_ins.config.get("flips_report_label_dist") is True

    def test_empty_results_returns_none_params(self) -> None:
        """aggregate_fit with no results returns (None, {})."""
        strategy = make_flips_fedavg(
            clients_per_round=4,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=_dummy_parameters(),
        )
        cids = [f"c{i}" for i in range(10)]
        manager = _make_client_manager(cids)
        for cid in cids:
            strategy._flips_registry.register(cid)

        strategy.configure_fit(1, _dummy_parameters(), manager)
        agg_params, metrics = strategy.aggregate_fit(1, [], [])
        assert agg_params is None

    def test_repr_contains_base_class_name(self) -> None:
        strategy = make_flips_fedavg(clients_per_round=5, initial_parameters=_dummy_parameters())
        assert "FedAvg" in repr(strategy)

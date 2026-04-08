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
"""FLIPS aggregation adapters for FedAvg, FedProx, and FedYogi.

FLIPS does not alter the core aggregation mathematics — it changes *which*
clients participate.  These adapters wrap the native Flower strategies and
inject FLIPS cluster-aware selection while delegating all aggregation to the
underlying strategy's ``aggregate_fit`` / ``aggregate_evaluate`` methods.

Supported base strategies
--------------------------
- :class:`~flwr.server.strategy.FedAvg`
- :class:`~flwr.server.strategy.FedProx`   (FedProx inherits from FedAvg)
- :class:`~flwr.server.strategy.FedYogi`
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, FedYogi

from flwr.server.flips.server.clustering import LabelDistributionClusterer
from flwr.server.flips.server.instrumentation import MetricsLogger, RoundMetrics
from flwr.server.flips.server.metadata_registry import MetadataRegistry
from flwr.server.flips.server.selector import FlipsSelector
from flwr.server.flips.server.straggler import StragglerTracker

# Metadata key sent via FitIns.config to instruct clients to report their
# label distribution in FitRes.metrics.
_LABEL_DIST_REQUEST_KEY = "flips_report_label_dist"
_TRAIN_TIME_KEY = "flips_train_time_s"
_STRAGGLER_SIMULATED_KEY = "flips_simulated_straggler"

# Prefix used for per-label metric keys in FitRes.metrics
_LABEL_DIST_PREFIX = "flips_ld_"
_NUM_SAMPLES_KEY = "flips_num_samples"


def _extract_label_dist_from_metrics(
    metrics: Dict[str, Scalar],
) -> Tuple[Dict[str, float], int]:
    """Parse label-distribution and sample count from ``FitRes.metrics``.

    The client encodes each label as ``flips_ld_<label> = probability`` and
    reports ``flips_num_samples = N``.

    Parameters
    ----------
    metrics:
        ``FitRes.metrics`` dict from a client.

    Returns
    -------
    (label_distribution, num_samples)
    """
    label_dist: Dict[str, float] = {}
    for key, val in metrics.items():
        if key.startswith(_LABEL_DIST_PREFIX):
            label = key[len(_LABEL_DIST_PREFIX):]
            label_dist[label] = float(val)
    num_samples = int(metrics.get(_NUM_SAMPLES_KEY, 0))
    return label_dist, num_samples


class FlipsStrategyMixin:
    """Mixin that injects FLIPS cluster-aware selection into any FedAvg-style strategy.

    Intended to be used via :func:`make_flips_strategy` below.  Not typically
    instantiated directly.

    FLIPS execution flow per round
    --------------------------------
    configure_fit (round R):
      1. First round: issue ``flips_report_label_dist = 1`` so every client
         reports its label histogram in FitRes.metrics.
      2. Compute (or reuse) cluster assignments via
         :class:`~flwr.server.flips.server.clustering.LabelDistributionClusterer`.
      3. Determine overprovisioning extra count from
         :class:`~flwr.server.flips.server.straggler.StragglerTracker`.
      4. Select clients via
         :class:`~flwr.server.flips.server.selector.FlipsSelector`.
      5. Record selections in the
         :class:`~flwr.server.flips.server.metadata_registry.MetadataRegistry`.

    aggregate_fit (round R):
      1. Parse label distributions from round-1 responses and update registry.
      2. Recluster if ``recluster_every`` rounds have elapsed.
      3. Record straggler outcomes in
         :class:`~flwr.server.flips.server.straggler.StragglerTracker`.
      4. Update per-client metadata.
      5. Invoke the parent strategy's ``aggregate_fit`` for model aggregation.
      6. Emit :class:`~flwr.server.flips.server.instrumentation.RoundMetrics`.
    """

    # These are declared here for type checkers; concrete values are set in
    # the generated subclass __init__ by make_flips_strategy().
    _flips_registry: MetadataRegistry
    _flips_clusterer: LabelDistributionClusterer
    _flips_selector: FlipsSelector
    _flips_straggler: StragglerTracker
    _flips_logger: MetricsLogger
    _flips_clients_per_round: int
    _flips_recluster_every: int
    _flips_initial_cluster_round: int
    _flips_request_label_dist_every: int
    _flips_round_start_times: Dict[int, float]
    _flips_round_selected: Dict[int, List[str]]
    _flips_round_per_cluster: Dict[int, Dict[int, List[str]]]
    _flips_overprovisioned_round: Dict[int, bool]
    _flips_clustered: bool

    def _flips_configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """FLIPS-aware implementation of ``configure_fit``."""
        all_proxies = client_manager.all()

        # Register any newly seen clients
        for cid in all_proxies:
            self._flips_registry.register(cid)

        # Determine whether to request label distributions
        request_labels = (
            server_round == 1
            or not self._flips_clustered
            or server_round % self._flips_request_label_dist_every == 0
        )

        # Build base config from parent's on_fit_config_fn if available
        base_config: Dict[str, Scalar] = {}
        _parent_on_fit = getattr(self, "on_fit_config_fn", None)
        if _parent_on_fit is not None:
            base_config = _parent_on_fit(server_round)

        if request_labels:
            base_config[_LABEL_DIST_REQUEST_KEY] = True

        # Cluster if needed
        if not self._flips_clustered or server_round % self._flips_recluster_every == 0:
            self._flips_recluster()

        # Determine overprovisioning
        extra = self._flips_straggler.extra_clients_needed(self._flips_clients_per_round)
        total_request = self._flips_clients_per_round + extra
        overprovisioned = extra > 0

        available_ids: FrozenSet[str] = frozenset(all_proxies.keys())

        if overprovisioned:
            affected = self._flips_straggler.last_affected_clusters()
            selected_ids, per_cluster = self._flips_selector.select_overprovision(
                clients_per_round=total_request,
                straggler_rate=self._flips_straggler.estimated_straggler_rate(),
                affected_clusters=affected if affected else None,
                available_ids=available_ids,
            )
        else:
            selected_ids, per_cluster = self._flips_selector.select(
                clients_per_round=total_request,
                available_ids=available_ids,
            )

        # Record selections
        for cid in selected_ids:
            self._flips_registry.record_selection(cid, server_round)

        self._flips_round_start_times[server_round] = time.monotonic()
        self._flips_round_selected[server_round] = list(selected_ids)
        self._flips_round_per_cluster[server_round] = per_cluster
        self._flips_overprovisioned_round[server_round] = overprovisioned

        # Build (proxy, FitIns) pairs for the selected clients only
        fit_ins = FitIns(parameters, base_config)
        results = [
            (all_proxies[cid], fit_ins)
            for cid in selected_ids
            if cid in all_proxies
        ]
        return results

    def _flips_aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """FLIPS post-processing hook for ``aggregate_fit``."""
        # Parse label distributions from results
        for proxy, fit_res in results:
            if _LABEL_DIST_PREFIX + "0" in fit_res.metrics or any(
                k.startswith(_LABEL_DIST_PREFIX) for k in fit_res.metrics
            ):
                ld, ns = _extract_label_dist_from_metrics(fit_res.metrics)
                if ld:
                    self._flips_registry.update_label_distribution(
                        proxy.cid, ld, ns
                    )

        # Recluster when we have new distributions
        if server_round == self._flips_initial_cluster_round:
            self._flips_recluster()

        # Track train time and stragglers
        selected = self._flips_round_selected.get(server_round, [])
        completed = [proxy.cid for proxy, _ in results]

        for proxy, fit_res in results:
            train_time = float(fit_res.metrics.get(_TRAIN_TIME_KEY, 0.0))
            self._flips_registry.record_fit_result(
                proxy.cid, train_time, was_straggler=False
            )

        straggler_cids = set(selected) - set(completed)
        for cid in straggler_cids:
            self._flips_registry.record_fit_result(cid, 0.0, was_straggler=True)

        # Record in straggler tracker
        client_cluster_map = {
            cid: (meta.cluster_id if meta else -1)
            for cid in selected
            if (meta := self._flips_registry.get(cid)) is not None
        }
        self._flips_straggler.record_round(
            server_round=server_round,
            selected=selected,
            completed=completed,
            client_cluster_map=client_cluster_map,
        )

        return results, failures  # signal to parent to proceed with aggregation

    def _flips_emit_metrics(
        self,
        server_round: int,
        aggregated_parameters: Optional[Parameters],
        fit_metrics: Dict[str, Scalar],
        test_loss: Optional[float],
        test_metrics: Dict[str, Scalar],
    ) -> None:
        """Build and record :class:`RoundMetrics` for this round."""
        selected = self._flips_round_selected.get(server_round, [])
        per_cluster_raw = self._flips_round_per_cluster.get(server_round, {})
        overprovisioned = self._flips_overprovisioned_round.get(server_round, False)

        # Per-cluster *completed* counts
        completed_set = {
            m.server_round == server_round
            for m in self._flips_straggler.history()
        }
        # Simpler: get from the straggler tracker's last record
        history = self._flips_straggler.history()
        if history and history[-1].server_round == server_round:
            completed = history[-1].completed
        else:
            completed = selected

        per_cluster_completed: Dict[int, List[str]] = {}
        registry_snap = self._flips_registry.all()
        for cid in completed:
            meta = registry_snap.get(cid)
            cluster_id = meta.cluster_id if meta else -1
            per_cluster_completed.setdefault(cluster_id, []).append(cid)

        round_metrics = MetricsLogger.build_from_round(
            server_round=server_round,
            selected=selected,
            completed=completed,
            per_cluster_completed=per_cluster_completed,
            straggler_rate_estimate=self._flips_straggler.estimated_straggler_rate(),
            overprovisioned=overprovisioned,
            test_loss=test_loss,
            test_metrics=dict(test_metrics),
            fit_metrics=dict(fit_metrics),
        )
        self._flips_logger.record(round_metrics)

    def _flips_recluster(self) -> None:
        """Run clustering with current label distributions from the registry."""
        all_meta = self._flips_registry.all()
        has_dist = {
            cid: meta.label_distribution
            for cid, meta in all_meta.items()
            if meta.label_distribution
        }
        if not has_dist:
            return

        result = self._flips_clusterer.fit(has_dist)
        for cid, cluster_id in result.client_cluster_map.items():
            self._flips_registry.update_cluster(cid, cluster_id)
        self._flips_clustered = True


def make_flips_strategy(
    base_strategy_cls: type,
    clients_per_round: int,
    registry: Optional[MetadataRegistry] = None,
    clusterer: Optional[LabelDistributionClusterer] = None,
    straggler_tracker: Optional[StragglerTracker] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    recluster_every: int = 5,
    request_label_dist_every: int = 10,
    initial_cluster_round: int = 1,
    selector_seed: Optional[int] = 42,
    log_path: Optional[Path] = None,
    **base_kwargs,
) -> "FlipsStrategyMixin":
    """Dynamically compose a FLIPS-enabled strategy from any FedAvg-derived class.

    The returned instance is both an instance of ``base_strategy_cls`` and of
    :class:`FlipsStrategyMixin`.

    Parameters
    ----------
    base_strategy_cls:
        A FedAvg-style Flower strategy class (e.g. ``FedAvg``, ``FedProx``,
        ``FedYogi``).
    clients_per_round:
        Desired number of clients per training round (before overprovisioning).
    registry:
        Pre-created registry.  A new one is created when ``None``.
    clusterer:
        Pre-created clusterer.  A new default one is created when ``None``.
    straggler_tracker:
        Pre-created tracker.  A new default one is created when ``None``.
    metrics_logger:
        Pre-created logger.  A new one (writing to ``log_path``) is created
        when ``None``.
    recluster_every:
        Recluster every N rounds.
    request_label_dist_every:
        Ask clients to re-report label distributions every N rounds.
    initial_cluster_round:
        The round after which the first cluster computation runs using the
        freshly collected label distributions from round 1.
    selector_seed:
        Seed for deterministic selection.
    log_path:
        Optional JSONL log file path forwarded to :class:`MetricsLogger`.
    **base_kwargs:
        Constructor keyword arguments forwarded to ``base_strategy_cls``.

    Returns
    -------
    An instance of a dynamically created class combining
    ``FlipsStrategyMixin`` and ``base_strategy_cls``.
    """

    _registry = registry or MetadataRegistry()
    _clusterer = clusterer or LabelDistributionClusterer()
    _tracker = straggler_tracker or StragglerTracker()
    _logger = metrics_logger or MetricsLogger(log_path=log_path)
    _selector = FlipsSelector(registry=_registry, seed=selector_seed)

    class FlipsStrategy(FlipsStrategyMixin, base_strategy_cls):  # type: ignore[valid-type]
        """Auto-generated FLIPS strategy wrapping ``base_strategy_cls``."""

        def __init__(self) -> None:
            # Initialise the parent strategy (FedAvg / FedProx / FedYogi)
            super().__init__(**base_kwargs)

            # FLIPS state
            self._flips_registry = _registry
            self._flips_clusterer = _clusterer
            self._flips_selector = _selector
            self._flips_straggler = _tracker
            self._flips_logger = _logger
            self._flips_clients_per_round = clients_per_round
            self._flips_recluster_every = recluster_every
            self._flips_request_label_dist_every = request_label_dist_every
            self._flips_initial_cluster_round = initial_cluster_round
            self._flips_round_start_times: Dict[int, float] = {}
            self._flips_round_selected: Dict[int, List[str]] = {}
            self._flips_round_per_cluster: Dict[int, Dict[int, List[str]]] = {}
            self._flips_overprovisioned_round: Dict[int, bool] = {}
            self._flips_clustered: bool = False

        def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
        ) -> List[Tuple[ClientProxy, FitIns]]:
            """FLIPS cluster-aware configure_fit."""
            return self._flips_configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """FLIPS post-processing + parent aggregation."""
            # Run FLIPS side-effects (register label dists, straggler tracking)
            self._flips_aggregate_fit(server_round, results, failures)

            # Delegate actual aggregation to base strategy
            aggregated_params, fit_metrics = super().aggregate_fit(
                server_round, results, failures
            )

            # Server-side evaluation (inherits parent evaluate())
            test_loss: Optional[float] = None
            test_metrics: Dict[str, Scalar] = {}
            eval_res = self.evaluate(server_round, aggregated_params or Parameters([], ""))
            if eval_res is not None:
                test_loss, test_metrics = eval_res

            self._flips_emit_metrics(
                server_round,
                aggregated_params,
                fit_metrics,
                test_loss,
                test_metrics,
            )
            return aggregated_params, fit_metrics

        def __repr__(self) -> str:
            return (
                f"FlipsStrategy(base={base_strategy_cls.__name__}, "
                f"clients_per_round={clients_per_round})"
            )

    return FlipsStrategy()


# ------------------------------------------------------------------ #
# Convenience factory functions                                         #
# ------------------------------------------------------------------ #


def make_flips_fedavg(clients_per_round: int, **kwargs) -> "FlipsStrategyMixin":
    """Create a FLIPS-enabled FedAvg strategy."""
    return make_flips_strategy(FedAvg, clients_per_round=clients_per_round, **kwargs)


def make_flips_fedprox(
    clients_per_round: int, proximal_mu: float = 0.01, **kwargs
) -> "FlipsStrategyMixin":
    """Create a FLIPS-enabled FedProx strategy.

    Parameters
    ----------
    proximal_mu:
        Proximal term weight ``μ`` forwarded to ``FedProx``.
    """
    return make_flips_strategy(
        FedProx,
        clients_per_round=clients_per_round,
        proximal_mu=proximal_mu,
        **kwargs,
    )


def make_flips_fedyogi(clients_per_round: int, **kwargs) -> "FlipsStrategyMixin":
    """Create a FLIPS-enabled FedYogi strategy."""
    return make_flips_strategy(FedYogi, clients_per_round=clients_per_round, **kwargs)

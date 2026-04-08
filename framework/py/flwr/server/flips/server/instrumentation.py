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
"""FLIPS round-level metrics and structured logging.

Accumulates per-round statistics and emits structured JSON-serialisable logs
that can be consumed by analysis scripts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """All server-observable metrics for a single federated round.

    Parameters
    ----------
    server_round:
        Round number (1-based).
    selected_clients:
        IDs of all clients asked to train this round (including extras
        selected for overprovisioning).
    completed_clients:
        IDs of clients that returned a valid update before the cutoff.
    straggler_clients:
        IDs of clients classified as stragglers / timed-out.
    selected_clusters:
        Distinct cluster IDs represented in ``selected_clients``.
    per_cluster_counts:
        Mapping ``cluster_id -> count`` of *completed* clients per cluster.
    effective_participation:
        ``len(completed_clients) / len(selected_clients)`` in (0, 1].
    test_loss:
        Global test loss after aggregation (``None`` if evaluate_fn not set).
    test_metrics:
        Any additional evaluation metrics returned by ``evaluate_fn``.
    fit_metrics:
        Aggregated fit metrics returned by the strategy's
        ``fit_metrics_aggregation_fn``.
    straggler_rate_estimate:
        Running straggler rate estimate at the start of this round.
    overprovisioned:
        ``True`` if extra clients were selected to compensate for expected
        stragglers.
    """

    server_round: int
    selected_clients: List[str] = field(default_factory=list)
    completed_clients: List[str] = field(default_factory=list)
    straggler_clients: List[str] = field(default_factory=list)
    selected_clusters: List[int] = field(default_factory=list)
    per_cluster_counts: Dict[int, int] = field(default_factory=dict)
    effective_participation: float = 0.0
    test_loss: Optional[float] = None
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    fit_metrics: Dict[str, Any] = field(default_factory=dict)
    straggler_rate_estimate: float = 0.0
    overprovisioned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain Python dict (JSON-compatible)."""
        d = asdict(self)
        # Convert integer dict keys to strings for JSON compatibility
        d["per_cluster_counts"] = {
            str(k): v for k, v in self.per_cluster_counts.items()
        }
        return d


class MetricsLogger:
    """Accumulate :class:`RoundMetrics` and optionally write a JSONL log file.

    Parameters
    ----------
    log_path:
        Optional path to a JSONL (one JSON object per line) log file.
        The file is created (and its parent directories) on first write.
        When ``None`` logs are only emitted via the ``logging`` module.
    log_level:
        Python logging level used for inline logging.  Defaults to
        ``logging.INFO``.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        log_level: int = logging.INFO,
    ) -> None:
        self._log_path = log_path
        self._log_level = log_level
        self._history: List[RoundMetrics] = []

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def record(self, metrics: RoundMetrics) -> None:
        """Append ``metrics`` to the history and flush to disk if configured.

        Parameters
        ----------
        metrics:
            Completed :class:`RoundMetrics` for a federated round.
        """
        self._history.append(metrics)
        self._log(metrics)
        if self._log_path is not None:
            self._write_jsonl(metrics)

    def history(self) -> List[RoundMetrics]:
        """Return a list of all recorded :class:`RoundMetrics` (oldest first)."""
        return list(self._history)

    def latest(self) -> Optional[RoundMetrics]:
        """Return the most recently recorded :class:`RoundMetrics`, or ``None``."""
        return self._history[-1] if self._history else None

    def summary(self) -> Dict[str, Any]:
        """Return aggregate summary statistics across all recorded rounds.

        Returns
        -------
        Dict[str, Any]
            Keys include ``total_rounds``, ``mean_effective_participation``,
            ``mean_straggler_rate``, ``total_stragglers``,
            ``cluster_coverage_per_round``.
        """
        if not self._history:
            return {"total_rounds": 0}

        participations = [m.effective_participation for m in self._history]
        straggler_counts = [len(m.straggler_clients) for m in self._history]
        cluster_coverages = [len(m.selected_clusters) for m in self._history]

        return {
            "total_rounds": len(self._history),
            "mean_effective_participation": sum(participations) / len(participations),
            "mean_straggler_rate": (
                sum(straggler_counts) / sum(len(m.selected_clients) for m in self._history)
                if sum(len(m.selected_clients) for m in self._history) > 0
                else 0.0
            ),
            "total_stragglers": sum(straggler_counts),
            "mean_cluster_coverage": sum(cluster_coverages) / len(cluster_coverages),
        }

    # ------------------------------------------------------------------ #
    # Static factory                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_from_round(
        server_round: int,
        selected: List[str],
        completed: List[str],
        per_cluster_completed: Dict[int, List[str]],
        straggler_rate_estimate: float,
        overprovisioned: bool,
        test_loss: Optional[float] = None,
        test_metrics: Optional[Dict[str, Any]] = None,
        fit_metrics: Optional[Dict[str, Any]] = None,
    ) -> RoundMetrics:
        """Construct a :class:`RoundMetrics` from raw round outcomes.

        Parameters
        ----------
        server_round:
            Current round number.
        selected:
            All selected client IDs (including extras).
        completed:
            Client IDs that returned valid results.
        per_cluster_completed:
            Mapping ``cluster_id -> [client_id]`` for *completed* clients.
        straggler_rate_estimate:
            Estimated straggler rate going into this round.
        overprovisioned:
            Whether extra clients were selected.
        test_loss:
            Global evaluation loss, if available.
        test_metrics:
            Additional evaluation metrics, if available.
        fit_metrics:
            Aggregated fit metrics, if available.

        Returns
        -------
        RoundMetrics
        """
        completed_set = set(completed)
        stragglers = [c for c in selected if c not in completed_set]
        selected_clusters = sorted(
            {clu for clu, members in per_cluster_completed.items() if members}
        )
        per_cluster_counts = {
            clu: len(members) for clu, members in per_cluster_completed.items()
        }
        participation = len(completed) / max(len(selected), 1)

        return RoundMetrics(
            server_round=server_round,
            selected_clients=list(selected),
            completed_clients=list(completed),
            straggler_clients=stragglers,
            selected_clusters=selected_clusters,
            per_cluster_counts=per_cluster_counts,
            effective_participation=participation,
            test_loss=test_loss,
            test_metrics=test_metrics or {},
            fit_metrics=fit_metrics or {},
            straggler_rate_estimate=straggler_rate_estimate,
            overprovisioned=overprovisioned,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _log(self, metrics: RoundMetrics) -> None:
        logger.log(
            self._log_level,
            "Round %d | selected=%d completed=%d stragglers=%d "
            "participation=%.2f clusters=%s",
            metrics.server_round,
            len(metrics.selected_clients),
            len(metrics.completed_clients),
            len(metrics.straggler_clients),
            metrics.effective_participation,
            metrics.selected_clusters,
        )

    def _write_jsonl(self, metrics: RoundMetrics) -> None:
        assert self._log_path is not None
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("FLIPS MetricsLogger: failed to write log: %s", exc)

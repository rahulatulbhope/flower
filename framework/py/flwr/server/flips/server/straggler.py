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
"""FLIPS straggler detection and overprovisioning tracker.

Tracks per-round straggler outcomes and maintains a running estimate of the
straggler rate used by :class:`~flwr.server.flips.server.selector.FlipsSelector`
to select extra clients in subsequent rounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class RoundStragglerRecord:
    """Straggler outcomes for a single federated round.

    Parameters
    ----------
    server_round:
        The round number this record belongs to.
    selected:
        Client IDs that were selected to participate.
    completed:
        Client IDs that returned a result before the cutoff.
    stragglers:
        Client IDs classified as stragglers (timeout/drop).
    affected_clusters:
        Distinct cluster IDs whose members were stragglers in this round.
    """

    server_round: int
    selected: List[str]
    completed: List[str]
    stragglers: List[str]
    affected_clusters: List[int]

    @property
    def straggler_rate(self) -> float:
        """Fraction of selected clients that straggled."""
        if not self.selected:
            return 0.0
        return len(self.stragglers) / len(self.selected)


class StragglerTracker:
    """Track straggler history and compute overprovisioning factor.

    Parameters
    ----------
    window:
        Number of most-recent rounds used in the rolling straggler-rate
        estimate.  Defaults to 5.
    ema_alpha:
        Exponential moving average factor for the global straggler rate
        estimate.  When ``None`` a simple windowed mean is used.
    min_overprovision:
        Minimum number of extra clients to add even when no history exists
        (default 0 = no forced minimum).
    max_overprovision_fraction:
        Cap on the extra fraction relative to the requested cohort size
        (default 1.0 = at most double the cohort).
    """

    def __init__(
        self,
        window: int = 5,
        ema_alpha: Optional[float] = None,
        min_overprovision: int = 0,
        max_overprovision_fraction: float = 1.0,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._ema_alpha = ema_alpha
        self._min_overprovision = min_overprovision
        self._max_overprovision_fraction = max_overprovision_fraction

        self._history: List[RoundStragglerRecord] = []
        self._ema_rate: float = 0.0

    # ------------------------------------------------------------------ #
    # Recording outcomes                                                    #
    # ------------------------------------------------------------------ #

    def record_round(
        self,
        server_round: int,
        selected: List[str],
        completed: List[str],
        client_cluster_map: Optional[Dict[str, int]] = None,
    ) -> RoundStragglerRecord:
        """Record the straggler outcome for a completed round.

        Parameters
        ----------
        server_round:
            Round number that just finished.
        selected:
            All client IDs that were asked to train this round.
        completed:
            Client IDs that returned a valid result before the cutoff.
        client_cluster_map:
            Optional mapping ``client_id -> cluster_id`` used to identify
            which clusters were affected by stragglers.  When ``None``
            no cluster-level information is stored.

        Returns
        -------
        RoundStragglerRecord
            The newly created record.
        """
        completed_set: Set[str] = set(completed)
        stragglers = [c for c in selected if c not in completed_set]

        affected: List[int] = []
        if client_cluster_map:
            affected = sorted(
                {client_cluster_map[c] for c in stragglers if c in client_cluster_map}
            )

        record = RoundStragglerRecord(
            server_round=server_round,
            selected=list(selected),
            completed=list(completed),
            stragglers=stragglers,
            affected_clusters=affected,
        )
        self._history.append(record)
        self._update_ema(record.straggler_rate)
        return record

    # ------------------------------------------------------------------ #
    # Rate estimation                                                       #
    # ------------------------------------------------------------------ #

    def estimated_straggler_rate(self) -> float:
        """Return the current estimate of the global straggler rate.

        Uses the EMA if ``ema_alpha`` was provided, otherwise a simple mean
        over the most recent ``window`` rounds.

        Returns
        -------
        float
            Estimated straggler rate in ``[0, 1]``.
        """
        if self._ema_alpha is not None:
            return self._ema_rate

        recent = self._history[-self._window :]
        if not recent:
            return 0.0
        return sum(r.straggler_rate for r in recent) / len(recent)

    def extra_clients_needed(self, clients_per_round: int) -> int:
        """Return the number of extra clients to select for overprovisioning.

        Parameters
        ----------
        clients_per_round:
            Base number of usable clients desired.

        Returns
        -------
        int
            Number of *additional* clients beyond ``clients_per_round``.
        """
        rate = self.estimated_straggler_rate()
        if rate <= 0.0:
            return self._min_overprovision
        if rate >= 1.0:
            # Entire cohort straggled — cap at max_overprovision_fraction
            return math.floor(clients_per_round * self._max_overprovision_fraction)

        # Extra needed to end up with ``clients_per_round`` completions:
        #   expected_completions = total * (1 - rate) >= clients_per_round
        #   => total >= clients_per_round / (1 - rate)
        extra = math.ceil(clients_per_round / (1.0 - rate)) - clients_per_round
        extra = max(extra, self._min_overprovision)
        cap = math.floor(clients_per_round * self._max_overprovision_fraction)
        return min(extra, cap)

    def last_affected_clusters(self) -> List[int]:
        """Return clusters that experienced stragglers in the most recent round."""
        if not self._history:
            return []
        return list(self._history[-1].affected_clusters)

    def history(self) -> List[RoundStragglerRecord]:
        """Return the full round history (oldest first)."""
        return list(self._history)

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _update_ema(self, rate: float) -> None:
        if self._ema_alpha is not None:
            self._ema_rate = (
                self._ema_alpha * rate + (1 - self._ema_alpha) * self._ema_rate
            )

    def __repr__(self) -> str:
        return (
            f"StragglerTracker("
            f"rounds={len(self._history)}, "
            f"rate_est={self.estimated_straggler_rate():.3f})"
        )

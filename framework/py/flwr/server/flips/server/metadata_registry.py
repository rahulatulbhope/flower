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
"""FLIPS client metadata registry.

Maintains per-client state across federated rounds: label distributions, cluster
assignments, participation counters, and straggler probability estimates.
Intentionally kept server-side only.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClientMetadata:
    """All server-side state for a single federated client.

    Parameters
    ----------
    client_id:
        Flower ``ClientProxy.cid`` string.
    num_samples:
        Number of local training samples last reported.
    label_distribution:
        Normalised histogram over class labels (values sum to ≤ 1).
        Keys are label names / integer indices cast to ``str``.
    cluster_id:
        Cluster index assigned by the last clustering run, ``-1`` if unknown.
    pick_count:
        Total number of rounds this client has been selected for training.
    last_seen_round:
        Most recent round in which the client was selected (0 = never).
    recent_train_time:
        Duration of the most recent local training step in seconds.
    is_straggler:
        Whether the client was classified as a straggler in the last round
        it participated in.
    straggler_probability:
        Exponential-moving-average estimate of the probability that this
        client will be a straggler (starts at 0.0).
    consecutive_failures:
        Number of consecutive rounds the client failed or timed out.
    """

    client_id: str
    num_samples: int = 0
    label_distribution: Dict[str, float] = field(default_factory=dict)
    cluster_id: int = -1
    pick_count: int = 0
    last_seen_round: int = 0
    recent_train_time: float = 0.0
    is_straggler: bool = False
    straggler_probability: float = 0.0
    consecutive_failures: int = 0

    # ------------------------------------------------------------------ #
    # Derived helpers                                                       #
    # ------------------------------------------------------------------ #

    def label_vector(self, num_classes: int) -> List[float]:
        """Return a dense label-distribution vector of length ``num_classes``.

        Labels not present in ``label_distribution`` are mapped to 0.0.

        Parameters
        ----------
        num_classes:
            Total number of classes in the federation.

        Returns
        -------
        List[float]
            Dense probability vector of length ``num_classes``.
        """
        vec = [0.0] * num_classes
        for label, prob in self.label_distribution.items():
            try:
                idx = int(label)
                if 0 <= idx < num_classes:
                    vec[idx] = prob
            except ValueError:
                pass
        return vec

    def update_straggler_ema(self, was_straggler: bool, alpha: float = 0.3) -> None:
        """Update the exponential moving average straggler probability.

        Parameters
        ----------
        was_straggler:
            Whether this client was a straggler in the most recent round.
        alpha:
            EMA smoothing factor in (0, 1).  Higher values weight recent
            observations more heavily.
        """
        observation = 1.0 if was_straggler else 0.0
        self.straggler_probability = (
            alpha * observation + (1 - alpha) * self.straggler_probability
        )
        self.is_straggler = was_straggler
        if was_straggler:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0


class MetadataRegistry:
    """Thread-safe server-side store for :class:`ClientMetadata`.

    All mutating operations acquire an internal lock so the registry
    is safe to use from multiple threads (e.g., Flower's server loop +
    a background metric-export thread).

    Parameters
    ----------
    straggler_ema_alpha:
        Smoothing factor forwarded to
        :meth:`ClientMetadata.update_straggler_ema`.
    """

    def __init__(self, straggler_ema_alpha: float = 0.3) -> None:
        self._lock = threading.Lock()
        self._clients: Dict[str, ClientMetadata] = {}
        self._alpha = straggler_ema_alpha

    # ------------------------------------------------------------------ #
    # Registration / lookup                                                #
    # ------------------------------------------------------------------ #

    def register(self, client_id: str) -> ClientMetadata:
        """Register a new client; return existing record if already present.

        Parameters
        ----------
        client_id:
            Unique string identifier (mirrors ``ClientProxy.cid``).

        Returns
        -------
        ClientMetadata
            The (potentially freshly created) record for ``client_id``.
        """
        with self._lock:
            if client_id not in self._clients:
                self._clients[client_id] = ClientMetadata(client_id=client_id)
            return self._clients[client_id]

    def get(self, client_id: str) -> Optional[ClientMetadata]:
        """Return metadata for ``client_id``, or ``None`` if unknown."""
        with self._lock:
            return self._clients.get(client_id)

    def all(self) -> Dict[str, ClientMetadata]:
        """Return a shallow copy of all registered client records."""
        with self._lock:
            return dict(self._clients)

    def ids(self) -> List[str]:
        """Return a list of all registered client IDs."""
        with self._lock:
            return list(self._clients.keys())

    # ------------------------------------------------------------------ #
    # Bulk updates                                                          #
    # ------------------------------------------------------------------ #

    def update_label_distribution(
        self,
        client_id: str,
        label_distribution: Dict[str, float],
        num_samples: int,
    ) -> None:
        """Overwrite the label distribution and sample count for a client.

        Registers the client if it has never been seen before.

        Parameters
        ----------
        client_id:
            Target client.
        label_distribution:
            Mapping from label name/index (str) → frequency / probability.
            Values may be raw counts; they are normalised to sum to 1.
        num_samples:
            Number of local training samples.
        """
        total = sum(label_distribution.values()) or 1
        normalised = {k: v / total for k, v in label_distribution.items()}
        with self._lock:
            meta = self._clients.setdefault(
                client_id, ClientMetadata(client_id=client_id)
            )
            meta.label_distribution = normalised
            meta.num_samples = num_samples

    def update_cluster(self, client_id: str, cluster_id: int) -> None:
        """Assign a cluster ID to an already-registered client."""
        with self._lock:
            if client_id in self._clients:
                self._clients[client_id].cluster_id = cluster_id

    def record_selection(self, client_id: str, server_round: int) -> None:
        """Increment pick counter and update last-seen round.

        Parameters
        ----------
        client_id:
            Client that was selected for training in ``server_round``.
        server_round:
            Current federated round number.
        """
        with self._lock:
            meta = self._clients.get(client_id)
            if meta is not None:
                meta.pick_count += 1
                meta.last_seen_round = server_round

    def record_fit_result(
        self,
        client_id: str,
        train_time: float,
        was_straggler: bool,
    ) -> None:
        """Update timing and straggler EMA after receiving a fit result.

        Parameters
        ----------
        client_id:
            Client that returned a fit result.
        train_time:
            Wall-clock training duration in seconds.
        was_straggler:
            ``True`` if the result arrived after the round cutoff or was
            considered a straggler by the server.
        """
        with self._lock:
            meta = self._clients.get(client_id)
            if meta is not None:
                meta.recent_train_time = train_time
                meta.update_straggler_ema(was_straggler, alpha=self._alpha)

    def bulk_update_labels(
        self,
        updates: Dict[str, tuple],  # client_id -> (label_dist, num_samples)
    ) -> None:
        """Apply label-distribution updates for multiple clients at once.

        Parameters
        ----------
        updates:
            Mapping from ``client_id`` to a tuple of
            ``(label_distribution: Dict[str, float], num_samples: int)``.
        """
        for client_id, (label_dist, num_samples) in updates.items():
            self.update_label_distribution(client_id, label_dist, num_samples)

    # ------------------------------------------------------------------ #
    # Queries                                                               #
    # ------------------------------------------------------------------ #

    def clients_by_cluster(self, cluster_id: int) -> List[str]:
        """Return IDs of all clients assigned to ``cluster_id``."""
        with self._lock:
            return [
                cid
                for cid, meta in self._clients.items()
                if meta.cluster_id == cluster_id
            ]

    def unclustered_clients(self) -> List[str]:
        """Return IDs of clients that have not yet been assigned a cluster."""
        with self._lock:
            return [
                cid
                for cid, meta in self._clients.items()
                if meta.cluster_id == -1
            ]

    def cluster_ids(self) -> List[int]:
        """Return a sorted list of distinct cluster IDs (excluding -1)."""
        with self._lock:
            return sorted(
                {meta.cluster_id for meta in self._clients.values()} - {-1}
            )

    def __len__(self) -> int:
        """Return the number of registered clients."""
        with self._lock:
            return len(self._clients)

    def __repr__(self) -> str:
        return (
            f"MetadataRegistry(n_clients={len(self)}, "
            f"n_clusters={len(self.cluster_ids())})"
        )

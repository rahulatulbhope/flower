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
"""FLIPS server-side label-distribution clustering.

Clusters federated clients by their label-distribution vectors using k-means,
with optional automatic k selection via the Davies-Bouldin index.  All
computation is server-side; only the aggregate histogram (not raw samples)
is used.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """Statistics for a single cluster after a clustering run.

    Parameters
    ----------
    cluster_id:
        Integer index of this cluster (0-based).
    member_ids:
        Sorted list of client IDs assigned to this cluster.
    centroid:
        Mean label-distribution vector for this cluster.
    intra_cluster_variance:
        Mean squared distance from each member to the centroid.
    """

    cluster_id: int
    member_ids: List[str]
    centroid: List[float]
    intra_cluster_variance: float


@dataclass
class ClusteringResult:
    """Output of a single clustering run.

    Parameters
    ----------
    client_cluster_map:
        Mapping from ``client_id`` to assigned ``cluster_id``.
    summaries:
        Per-cluster statistics indexed by ``cluster_id``.
    k:
        Number of clusters used.
    davies_bouldin_score:
        DB index of this partition (lower is better; ``None`` when k == 1).
    """

    client_cluster_map: Dict[str, int]
    summaries: Dict[int, ClusterSummary]
    k: int
    davies_bouldin_score: Optional[float]


# --------------------------------------------------------------------------- #
# Internal maths helpers                                                        #
# --------------------------------------------------------------------------- #

def _davies_bouldin(
    X: NDArray[np.float64],
    labels: NDArray[np.int32],
    centroids: NDArray[np.float64],
) -> float:
    """Compute the Davies-Bouldin index for a clustering.

    Parameters
    ----------
    X:
        Feature matrix of shape (n_samples, n_features).
    labels:
        Integer cluster labels of length n_samples.
    centroids:
        Centroid matrix of shape (k, n_features).

    Returns
    -------
    float
        Davies-Bouldin index (lower is better, >= 0).
    """
    k = len(centroids)
    if k <= 1:
        return 0.0

    # Scatter: mean distance of cluster members to their centroid
    scatter = np.zeros(k)
    for i in range(k):
        members = X[labels == i]
        if len(members) == 0:
            continue
        scatter[i] = float(np.mean(np.linalg.norm(members - centroids[i], axis=1)))

    db_sum = 0.0
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j:
                continue
            dist_ij = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist_ij < 1e-10:
                dist_ij = 1e-10
            ratios.append((scatter[i] + scatter[j]) / dist_ij)
        db_sum += max(ratios)

    return db_sum / k


def _kmeans(
    X: NDArray[np.float64],
    k: int,
    max_iter: int = 300,
    seed: Optional[int] = None,
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Pure-NumPy k-means (no scikit-learn dependency).

    Uses k-means++ seeding for deterministic, high-quality initialisation.

    Parameters
    ----------
    X:
        Data matrix of shape (n_samples, n_features).
    k:
        Number of clusters.
    max_iter:
        Maximum number of Lloyd iterations.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    labels:
        Integer cluster assignments of length n_samples.
    centroids:
        Centroid matrix of shape (k, n_features).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)  # Can't have more clusters than samples

    # k-means++ initialisation
    centroids: List[NDArray[np.float64]] = [X[rng.integers(n)].copy()]
    for _ in range(1, k):
        dist_sq = np.array(
            [
                min(float(np.linalg.norm(x - c) ** 2) for c in centroids)
                for x in X
            ]
        )
        dist_sq_sum = dist_sq.sum()
        if dist_sq_sum < 1e-12:
            # All points essentially the same; pick arbitrarily
            centroids.append(X[rng.integers(n)].copy())
        else:
            probs = dist_sq / dist_sq_sum
            idx = rng.choice(n, p=probs)
            centroids.append(X[idx].copy())

    centroid_arr = np.array(centroids)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroid_arr[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update centroids
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroid_arr[j] = members.mean(axis=0)

    return labels, centroid_arr


# --------------------------------------------------------------------------- #
# Main clusterer                                                                #
# --------------------------------------------------------------------------- #

class LabelDistributionClusterer:
    """Cluster federated clients by their label-distribution vectors.

    Parameters
    ----------
    k:
        Fixed number of clusters.  Set to ``None`` to enable automatic
        selection via the Davies-Bouldin index.
    k_min:
        Minimum k to consider when ``k`` is ``None``.  Defaults to 2.
    k_max:
        Maximum k to consider when ``k`` is ``None``.  Defaults to 10.
        Capped internally at ``n_clients - 1``.
    num_classes:
        Total number of label classes in the federation.  Used to build
        dense feature vectors.  When ``None`` it is inferred lazily from
        the input distributions.
    seed:
        Random seed forwarded to the k-means initialiser.
    max_iter:
        Maximum k-means Lloyd iterations per run.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        k_min: int = 2,
        k_max: int = 10,
        num_classes: Optional[int] = None,
        seed: Optional[int] = 42,
        max_iter: int = 300,
    ) -> None:
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._k_min = k_min
        self._k_max = k_max
        self._num_classes = num_classes
        self._seed = seed
        self._max_iter = max_iter

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        client_distributions: Dict[str, Dict[str, float]],
    ) -> ClusteringResult:
        """Cluster clients given their label-distribution dictionaries.

        Parameters
        ----------
        client_distributions:
            Mapping ``client_id -> {label_str -> probability}``.
            Distributions are normalised internally.

        Returns
        -------
        ClusteringResult
            Full clustering outcome including per-cluster summaries.

        Raises
        ------
        ValueError
            If fewer than two clients are provided.
        """
        if len(client_distributions) < 1:
            raise ValueError("Need at least one client to cluster.")

        client_ids = sorted(client_distributions.keys())
        X, num_classes = self._build_feature_matrix(client_ids, client_distributions)

        if len(client_ids) == 1:
            return self._single_client_result(client_ids[0], X)

        k = self._resolve_k(X)
        labels, centroids = _kmeans(X, k, max_iter=self._max_iter, seed=self._seed)

        db_score: Optional[float] = None
        if k > 1:
            db_score = _davies_bouldin(X, labels, centroids)

        return self._build_result(client_ids, X, labels, centroids, k, db_score)

    def refit(
        self,
        client_distributions: Dict[str, Dict[str, float]],
    ) -> ClusteringResult:
        """Alias for :meth:`fit`; provided for API clarity in re-clustering workflows."""
        return self.fit(client_distributions)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _build_feature_matrix(
        self,
        client_ids: List[str],
        distributions: Dict[str, Dict[str, float]],
    ) -> Tuple[NDArray[np.float64], int]:
        """Build a (n_clients, n_classes) feature matrix.

        Unknown labels are mapped via string-to-int cast.  Non-integer
        label strings are consistent-hashed to a position beyond the
        currently known integer range.
        """
        # Determine the label universe
        all_labels: set[str] = set()
        for dist in distributions.values():
            all_labels.update(dist.keys())

        int_labels: List[int] = []
        str_labels: List[str] = []
        for lbl in all_labels:
            try:
                int_labels.append(int(lbl))
            except ValueError:
                str_labels.append(lbl)

        max_int = max(int_labels) if int_labels else -1
        str_label_map: Dict[str, int] = {
            s: max_int + 1 + i for i, s in enumerate(sorted(str_labels))
        }

        inferred_classes = max_int + 1 + len(str_labels)
        num_classes = (
            self._num_classes
            if self._num_classes is not None
            else max(inferred_classes, 1)
        )

        X = np.zeros((len(client_ids), num_classes), dtype=np.float64)
        for row, cid in enumerate(client_ids):
            dist = distributions[cid]
            total = sum(dist.values()) or 1.0
            for label, count in dist.items():
                try:
                    idx = int(label)
                except ValueError:
                    idx = str_label_map.get(label, num_classes - 1)
                if 0 <= idx < num_classes:
                    X[row, idx] = count / total

        return X, num_classes

    def _resolve_k(self, X: NDArray[np.float64]) -> int:
        """Return the number of clusters to use for this run."""
        n = len(X)
        if self._k is not None:
            return min(self._k, n)

        k_min = max(2, self._k_min)
        k_max = min(self._k_max, n - 1)
        if k_max < k_min:
            return max(1, k_max)

        best_k = k_min
        best_db = math.inf

        for candidate_k in range(k_min, k_max + 1):
            labels, centroids = _kmeans(
                X, candidate_k, max_iter=self._max_iter, seed=self._seed
            )
            if len(set(labels.tolist())) < candidate_k:
                # Degenerate solution — fewer real clusters than requested
                continue
            db = _davies_bouldin(X, labels, centroids)
            logger.debug("k=%d DB=%.4f", candidate_k, db)
            if db < best_db:
                best_db = db
                best_k = candidate_k

        return best_k

    @staticmethod
    def _single_client_result(client_id: str, X: NDArray[np.float64]) -> ClusteringResult:
        centroid = X[0].tolist()
        summary = ClusterSummary(
            cluster_id=0,
            member_ids=[client_id],
            centroid=centroid,
            intra_cluster_variance=0.0,
        )
        return ClusteringResult(
            client_cluster_map={client_id: 0},
            summaries={0: summary},
            k=1,
            davies_bouldin_score=None,
        )

    @staticmethod
    def _build_result(
        client_ids: List[str],
        X: NDArray[np.float64],
        labels: NDArray[np.int32],
        centroids: NDArray[np.float64],
        k: int,
        db_score: Optional[float],
    ) -> ClusteringResult:
        """Construct :class:`ClusteringResult` from raw k-means outputs."""
        client_cluster_map: Dict[str, int] = {
            cid: int(labels[i]) for i, cid in enumerate(client_ids)
        }

        summaries: Dict[int, ClusterSummary] = {}
        for cluster_id in range(k):
            member_indices = [
                i for i, _ in enumerate(client_ids) if labels[i] == cluster_id
            ]
            member_ids = sorted(client_ids[i] for i in member_indices)
            centroid = centroids[cluster_id]

            if member_indices:
                member_rows = X[np.array(member_indices)]
                dists_sq = np.sum((member_rows - centroid) ** 2, axis=1)
                variance = float(dists_sq.mean())
            else:
                variance = 0.0

            summaries[cluster_id] = ClusterSummary(
                cluster_id=cluster_id,
                member_ids=member_ids,
                centroid=centroid.tolist(),
                intra_cluster_variance=variance,
            )

        return ClusteringResult(
            client_cluster_map=client_cluster_map,
            summaries=summaries,
            k=k,
            davies_bouldin_score=db_score,
        )

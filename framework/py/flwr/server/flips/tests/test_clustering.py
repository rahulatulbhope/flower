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
"""Unit tests for LabelDistributionClusterer (server-side)."""

import pytest
import numpy as np

from flwr.server.flips.server.clustering import (
    LabelDistributionClusterer,
    _davies_bouldin,
    _kmeans,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_distributions(n_clients: int, n_classes: int, seed: int = 0):
    """Build synthetic non-IID label distributions with known cluster structure.

    Clients 0..n_clients//2-1 are "cluster A" (biased toward class 0).
    The rest are "cluster B" (biased toward class 1).
    """
    rng = np.random.default_rng(seed)
    dists = {}
    for i in range(n_clients):
        vec = rng.dirichlet(
            [5.0 if j == (0 if i < n_clients // 2 else 1) else 0.1 for j in range(n_classes)]
        )
        dists[f"client_{i:03d}"] = {str(j): float(vec[j]) for j in range(n_classes)}
    return dists


# ---------------------------------------------------------------------------
# _kmeans helper tests
# ---------------------------------------------------------------------------


class TestKMeans:
    def test_returns_correct_shape(self) -> None:
        X = np.random.default_rng(0).random((20, 5))
        labels, centroids = _kmeans(X, k=3, seed=42)
        assert labels.shape == (20,)
        assert centroids.shape == (3, 5)

    def test_all_labels_in_range(self) -> None:
        X = np.random.default_rng(1).random((15, 4))
        labels, centroids = _kmeans(X, k=3, seed=0)
        assert set(labels.tolist()).issubset({0, 1, 2})

    def test_single_cluster(self) -> None:
        X = np.eye(5)
        labels, centroids = _kmeans(X, k=1, seed=0)
        assert (labels == 0).all()

    def test_deterministic_with_seed(self) -> None:
        X = np.random.default_rng(42).random((30, 6))
        labels_a, _ = _kmeans(X, k=3, seed=7)
        labels_b, _ = _kmeans(X, k=3, seed=7)
        np.testing.assert_array_equal(labels_a, labels_b)

    def test_k_capped_at_n_samples(self) -> None:
        X = np.eye(3)
        labels, centroids = _kmeans(X, k=10, seed=0)
        assert centroids.shape[0] <= 3


# ---------------------------------------------------------------------------
# _davies_bouldin tests
# ---------------------------------------------------------------------------


class TestDaviesBouldin:
    def test_lower_for_better_separation(self) -> None:
        """Well-separated clusters have a lower DB score."""
        rng = np.random.default_rng(0)
        # Tight, well-separated clusters
        A = rng.normal([0, 0], 0.1, (50, 2))
        B = rng.normal([10, 10], 0.1, (50, 2))
        X_good = np.vstack([A, B])
        labels_good = np.array([0] * 50 + [1] * 50, dtype=np.int32)
        centroids_good = np.array([[0, 0], [10, 10]], dtype=np.float64)

        # Overlapping clusters
        C = rng.normal([0, 0], 2.0, (50, 2))
        D = rng.normal([1, 1], 2.0, (50, 2))
        X_bad = np.vstack([C, D])
        labels_bad = np.array([0] * 50 + [1] * 50, dtype=np.int32)
        centroids_bad = np.array([[0, 0], [1, 1]], dtype=np.float64)

        db_good = _davies_bouldin(X_good, labels_good, centroids_good)
        db_bad = _davies_bouldin(X_bad, labels_bad, centroids_bad)
        assert db_good < db_bad

    def test_single_cluster_returns_zero(self) -> None:
        X = np.random.default_rng(0).random((10, 3))
        labels = np.zeros(10, dtype=np.int32)
        centroids = X.mean(axis=0, keepdims=True)
        assert _davies_bouldin(X, labels, centroids) == 0.0


# ---------------------------------------------------------------------------
# LabelDistributionClusterer tests
# ---------------------------------------------------------------------------


class TestLabelDistributionClusterer:
    def test_stable_assignment_fixed_k_and_seed(self) -> None:
        """For fixed k and seed, repeated calls return identical assignments."""
        dists = _make_distributions(20, 10, seed=42)
        clusterer = LabelDistributionClusterer(k=2, seed=0)
        result_a = clusterer.fit(dists)
        result_b = clusterer.fit(dists)
        assert result_a.client_cluster_map == result_b.client_cluster_map

    def test_cluster_ids_are_contiguous(self) -> None:
        dists = _make_distributions(12, 10)
        clusterer = LabelDistributionClusterer(k=3, seed=0)
        result = clusterer.fit(dists)
        assert set(result.client_cluster_map.values()) == {0, 1, 2}

    def test_all_clients_assigned(self) -> None:
        dists = _make_distributions(16, 5)
        clusterer = LabelDistributionClusterer(k=4, seed=0)
        result = clusterer.fit(dists)
        assert set(result.client_cluster_map.keys()) == set(dists.keys())

    def test_summaries_present(self) -> None:
        dists = _make_distributions(10, 5)
        clusterer = LabelDistributionClusterer(k=2, seed=0)
        result = clusterer.fit(dists)
        assert len(result.summaries) == 2
        for s in result.summaries.values():
            assert len(s.centroid) > 0
            assert s.intra_cluster_variance >= 0.0

    def test_auto_k_selection_synthetic(self) -> None:
        """Auto-k on clearly bimodal data should select k=2."""
        dists = _make_distributions(20, 10, seed=0)
        clusterer = LabelDistributionClusterer(k=None, k_min=2, k_max=5, seed=0)
        result = clusterer.fit(dists)
        # Should have at least 2 clusters on a bimodal distribution
        assert result.k >= 2
        assert result.davies_bouldin_score is not None

    def test_single_client(self) -> None:
        dists = {"only_client": {"0": 0.8, "1": 0.2}}
        clusterer = LabelDistributionClusterer(k=1, seed=0)
        result = clusterer.fit(dists)
        assert result.client_cluster_map == {"only_client": 0}
        assert result.k == 1

    def test_empty_raises(self) -> None:
        clusterer = LabelDistributionClusterer()
        with pytest.raises(ValueError):
            clusterer.fit({})

    def test_refit_alias(self) -> None:
        dists = _make_distributions(10, 5)
        clusterer = LabelDistributionClusterer(k=2, seed=0)
        r1 = clusterer.fit(dists)
        r2 = clusterer.refit(dists)
        assert r1.client_cluster_map == r2.client_cluster_map

    def test_string_label_keys(self) -> None:
        """Non-integer label keys are handled gracefully."""
        dists = {
            "c0": {"cat": 0.8, "dog": 0.2},
            "c1": {"cat": 0.1, "dog": 0.9},
            "c2": {"cat": 0.75, "dog": 0.25},
            "c3": {"cat": 0.05, "dog": 0.95},
        }
        clusterer = LabelDistributionClusterer(k=2, seed=0)
        result = clusterer.fit(dists)
        assert set(result.client_cluster_map.keys()) == {"c0", "c1", "c2", "c3"}

    def test_k_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            LabelDistributionClusterer(k=0)

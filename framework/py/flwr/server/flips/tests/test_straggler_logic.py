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
"""Unit tests for StragglerTracker."""

import math

import pytest

from flwr.server.flips.server.straggler import StragglerTracker


class TestStragglerTracker:
    def test_initial_straggler_rate_zero(self) -> None:
        tracker = StragglerTracker()
        assert tracker.estimated_straggler_rate() == 0.0

    def test_record_round_no_stragglers(self) -> None:
        tracker = StragglerTracker()
        selected = ["c0", "c1", "c2"]
        completed = ["c0", "c1", "c2"]
        record = tracker.record_round(1, selected, completed)
        assert record.straggler_rate == 0.0
        assert tracker.estimated_straggler_rate() == 0.0

    def test_straggler_rate_computed_correctly(self) -> None:
        tracker = StragglerTracker()
        selected = ["c0", "c1", "c2", "c3"]
        completed = ["c0", "c1"]
        record = tracker.record_round(1, selected, completed)
        assert abs(record.straggler_rate - 0.5) < 1e-9

    def test_windowed_mean_rate(self) -> None:
        tracker = StragglerTracker(window=3)
        # Round 1: 0% stragglers
        tracker.record_round(1, ["a", "b"], ["a", "b"])
        # Round 2: 50% stragglers
        tracker.record_round(2, ["a", "b"], ["a"])
        # Round 3: 50% stragglers
        tracker.record_round(3, ["a", "b"], ["b"])
        # Window = 3, mean = (0 + 0.5 + 0.5) / 3
        expected = (0.0 + 0.5 + 0.5) / 3
        assert abs(tracker.estimated_straggler_rate() - expected) < 1e-9

    def test_window_only_uses_recent_rounds(self) -> None:
        tracker = StragglerTracker(window=2)
        # Old round with 100% stragglers
        tracker.record_round(1, ["a", "b"], [])
        # Two recent rounds with 0% stragglers
        tracker.record_round(2, ["a", "b"], ["a", "b"])
        tracker.record_round(3, ["a", "b"], ["a", "b"])
        # Window = 2 → only rounds 2 and 3 matter
        assert tracker.estimated_straggler_rate() == 0.0

    def test_ema_rate_update(self) -> None:
        tracker = StragglerTracker(ema_alpha=0.5)
        tracker.record_round(1, ["a", "b"], ["a"])  # 50% straggler
        rate = tracker.estimated_straggler_rate()
        # EMA: 0.5 * 0.5 + 0.5 * 0.0 = 0.25
        assert abs(rate - 0.25) < 1e-9

    def test_affected_clusters_identified(self) -> None:
        tracker = StragglerTracker()
        cluster_map = {"a": 0, "b": 1, "c": 2}
        tracker.record_round(1, ["a", "b", "c"], ["a"], client_cluster_map=cluster_map)
        affected = tracker.last_affected_clusters()
        assert sorted(affected) == [1, 2]

    def test_extra_clients_needed_zero_when_no_history(self) -> None:
        tracker = StragglerTracker()
        assert tracker.extra_clients_needed(10) == 0

    def test_extra_clients_needed_positive(self) -> None:
        tracker = StragglerTracker()
        tracker.record_round(1, ["a", "b", "c", "d"], ["a", "b"])  # 50%
        extra = tracker.extra_clients_needed(10)
        assert extra > 0

    def test_extra_clients_capped_by_max_fraction(self) -> None:
        tracker = StragglerTracker(max_overprovision_fraction=0.5)
        tracker.record_round(1, ["a"], [])  # 100% straggler (edge)
        extra = tracker.extra_clients_needed(10)
        assert extra <= math.floor(10 * 0.5)

    def test_history_grows_per_round(self) -> None:
        tracker = StragglerTracker()
        for i in range(5):
            tracker.record_round(i + 1, ["x"], ["x"])
        assert len(tracker.history()) == 5

    def test_last_affected_clusters_empty_initially(self) -> None:
        tracker = StragglerTracker()
        assert tracker.last_affected_clusters() == []

    def test_min_overprovision_enforced(self) -> None:
        tracker = StragglerTracker(min_overprovision=3)
        # No stragglers — extra should still be at least min_overprovision
        assert tracker.extra_clients_needed(10) >= 3

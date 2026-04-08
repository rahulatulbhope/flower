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
"""Unit tests for LabelDistributionReporter (client-side)."""

import pytest

from flwr.server.flips.client.label_reporter import LabelDistributionReporter


class TestLabelDistributionReporter:
    """Tests for correct histogram computation and encoding."""

    def test_basic_histogram_integer_labels(self) -> None:
        """Standard integer labels produce a normalised histogram."""
        reporter = LabelDistributionReporter()
        labels = [0, 0, 0, 1, 1, 2]
        dist = reporter.compute(labels)

        assert set(dist.keys()) == {"0", "1", "2"}
        assert abs(dist["0"] - 3 / 6) < 1e-9
        assert abs(dist["1"] - 2 / 6) < 1e-9
        assert abs(dist["2"] - 1 / 6) < 1e-9

    def test_histogram_sums_to_one(self) -> None:
        """Normalised distribution sums to 1.0."""
        reporter = LabelDistributionReporter()
        labels = list(range(10)) * 3 + [0, 1]
        dist = reporter.compute(labels)
        assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_single_class_client(self) -> None:
        """A client with only one class gets a distribution of {class: 1.0}."""
        reporter = LabelDistributionReporter()
        labels = [5, 5, 5, 5]
        dist = reporter.compute(labels)
        assert dist == {"5": 1.0}

    def test_missing_labels_are_absent(self) -> None:
        """Labels not present in the dataset are not included in the distribution."""
        reporter = LabelDistributionReporter(num_classes=10)
        labels = [0, 1]
        dist = reporter.compute(labels)
        # Only classes 0 and 1 appear
        assert set(dist.keys()) == {"0", "1"}

    def test_raw_counts_when_not_normalised(self) -> None:
        """``normalise=False`` returns raw counts."""
        reporter = LabelDistributionReporter(normalise=False)
        labels = [0, 0, 1]
        dist = reporter.compute(labels)
        assert dist["0"] == 2
        assert dist["1"] == 1

    def test_string_labels(self) -> None:
        """String labels are supported and preserve key identity."""
        reporter = LabelDistributionReporter()
        labels = ["cat", "cat", "dog", "bird"]
        dist = reporter.compute(labels)
        assert set(dist.keys()) == {"cat", "dog", "bird"}
        assert abs(dist["cat"] - 2 / 4) < 1e-9

    def test_class_names_mapping(self) -> None:
        """Integer labels are mapped to class names when provided."""
        reporter = LabelDistributionReporter(class_names={0: "cat", 1: "dog"})
        labels = [0, 0, 1]
        dist = reporter.compute(labels)
        assert "cat" in dist
        assert "dog" in dist

    def test_empty_input(self) -> None:
        """Empty label iterable returns an empty distribution."""
        reporter = LabelDistributionReporter()
        dist = reporter.compute([])
        assert dist == {}

    def test_encode_for_metrics_adds_prefix(self) -> None:
        """Encoded dict has correct prefix on all keys."""
        reporter = LabelDistributionReporter()
        dist = {"0": 0.6, "1": 0.4}
        encoded = reporter.encode_for_metrics(dist, prefix="flips_ld_")
        assert "flips_ld_0" in encoded
        assert "flips_ld_1" in encoded
        assert abs(encoded["flips_ld_0"] - 0.6) < 1e-9

    def test_full_report_pipeline(self) -> None:
        """full_report chains compute and encode correctly."""
        reporter = LabelDistributionReporter()
        labels = [0, 1, 1, 2]
        result = reporter.full_report(labels)
        assert all(k.startswith("flips_ld_") for k in result)
        assert abs(sum(result.values()) - 1.0) < 1e-9

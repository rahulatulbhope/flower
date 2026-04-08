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
"""FLIPS client-side label-distribution reporter.

Reports only an aggregate histogram over class labels — never raw training
examples — so server-side clustering can assign the client to a cluster
without violating data privacy.

The reporter is deliberately independent of any ML framework (PyTorch,
TensorFlow, etc.) and works with any dataset that yields integer or string
labels.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Union


class LabelDistributionReporter:
    """Compute and encode a label-histogram for FLIPS metadata reporting.

    Parameters
    ----------
    num_classes:
        Total number of classes in the federation.  When ``None`` the
        reported distribution is sparse and only contains observed labels.
    class_names:
        Optional mapping from integer index to human-readable class name.
        When provided, labels are reported under the class name string.
        Otherwise labels are encoded as ``str(int_index)``.
    normalise:
        When ``True`` (default), the histogram is normalised so that values
        sum to 1.0.  When ``False`` raw counts are returned.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        normalise: bool = True,
    ) -> None:
        self._num_classes = num_classes
        self._class_names = class_names or {}
        self._normalise = normalise

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        labels: Iterable[Union[int, str]],
    ) -> Dict[str, float]:
        """Compute a label distribution histogram from an iterable of labels.

        Parameters
        ----------
        labels:
            Iterable of integer or string labels from the local dataset.
            Each element corresponds to one training example.

        Returns
        -------
        Dict[str, float]
            Mapping from label key (str) to normalised frequency (or raw
            count when ``normalise=False``).
        """
        counter: Counter = Counter()
        for lbl in labels:
            key = self._label_key(lbl)
            counter[key] += 1

        if not counter:
            return {}

        if self._normalise:
            total = sum(counter.values())
            return {k: v / total for k, v in counter.items()}
        return dict(counter)

    def compute_from_targets(
        self,
        targets: Iterable[Union[int, str]],
    ) -> Dict[str, float]:
        """Alias for :meth:`compute`; provided for PyTorch Dataset compatibility."""
        return self.compute(targets)

    def encode_for_metrics(
        self,
        distribution: Dict[str, float],
        prefix: str = "flips_ld_",
    ) -> Dict[str, float]:
        """Encode a distribution dict so it can be included in ``FitRes.metrics``.

        Flower's ``metrics`` type is ``Dict[str, Scalar]`` where scalar
        values are ``int | float | str | bytes | bool``.  This method
        prepends ``prefix`` to each key so the server can extract FLIPS
        label distribution entries reliably.

        Parameters
        ----------
        distribution:
            Mapping produced by :meth:`compute`.
        prefix:
            Prefix to prepend to every key.  Must match the server-side
            extraction prefix (``"flips_ld_"`` by default).

        Returns
        -------
        Dict[str, float]
            Ready-to-merge dict for ``FitRes.metrics``.
        """
        return {f"{prefix}{k}": v for k, v in distribution.items()}

    def full_report(
        self,
        labels: Iterable[Union[int, str]],
        prefix: str = "flips_ld_",
    ) -> Dict[str, float]:
        """Compute the distribution and return it already encoded for metrics.

        Combines :meth:`compute` and :meth:`encode_for_metrics` in a single
        call.

        Parameters
        ----------
        labels:
            Iterable of labels from the local dataset.
        prefix:
            Key prefix for the FLIPS label-distribution entries.

        Returns
        -------
        Dict[str, float]
        """
        dist = self.compute(labels)
        return self.encode_for_metrics(dist, prefix=prefix)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _label_key(self, label: Union[int, str]) -> str:
        """Convert a label to the string key used in the distribution dict."""
        if isinstance(label, str):
            return label
        # Integer label: look up class name or use str representation
        return self._class_names.get(int(label), str(label))

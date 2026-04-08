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
"""FLIPS standard Flower NumPy client.

A lightweight :class:`~flwr.client.NumPyClient` subclass that:
- Trains locally using FedAvg or FedProx objectives.
- Reports label-distribution metadata when requested by the server.
- Measures and reports local training time.

Everything ML-framework-specific is delegated to user-supplied callables,
keeping this class framework-agnostic.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from flwr.client import NumPyClient
from flwr.common import Config, NDArrays, Scalar

from flwr.server.flips.client.label_reporter import LabelDistributionReporter
from flwr.server.flips.client.straggler_sim import StragglerSimulator

# Keys defined here must match the server-side constants in aggregation.py
_LABEL_DIST_REQUEST_KEY = "flips_report_label_dist"
_LABEL_DIST_PREFIX = "flips_ld_"
_TRAIN_TIME_KEY = "flips_train_time_s"
_NUM_SAMPLES_KEY = "flips_num_samples"

# FedProx config key sent by the server
_PROXIMAL_MU_KEY = "proximal_mu"


class FlipsNumPyClient(NumPyClient):
    """FLIPS-aware Flower NumPy client.

    Parameters
    ----------
    get_parameters_fn:
        Callable ``() -> NDArrays`` returning the current model parameters.
    set_parameters_fn:
        Callable ``(NDArrays) -> None`` loading parameters into the model.
    train_fn:
        Callable ``(config: Config) -> (num_samples, metrics)`` that runs
        local training and returns the number of samples used plus a metrics
        dict.  After this call, ``get_parameters_fn()`` must return the
        updated parameters.
    evaluate_fn:
        Callable ``(config: Config) -> (loss, num_samples, metrics)``
        that evaluates the model on the local test set.
    label_iterable:
        An iterable (e.g., a list of integer labels) used by the
        :class:`~flwr.server.flips.client.label_reporter.LabelDistributionReporter`
        to compute the label histogram.  When ``None`` label reporting is
        disabled even if the server requests it.
    num_classes:
        Forwarded to :class:`LabelDistributionReporter`.
    class_names:
        Optional integer → name mapping forwarded to
        :class:`LabelDistributionReporter`.
    straggler_simulator:
        Optional :class:`~flwr.server.flips.client.straggler_sim.StragglerSimulator`
        for experiment/testing purposes.  When provided, it may inject an
        artificial delay or silently drop the response.
    """

    def __init__(
        self,
        get_parameters_fn: Callable[[], NDArrays],
        set_parameters_fn: Callable[[NDArrays], None],
        train_fn: Callable[[Config], Tuple[int, Dict[str, Scalar]]],
        evaluate_fn: Callable[[Config], Tuple[float, int, Dict[str, Scalar]]],
        label_iterable: Optional[Iterable[Union[int, str]]] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        straggler_simulator: Optional["StragglerSimulator"] = None,
    ) -> None:
        super().__init__()
        self._get_params = get_parameters_fn
        self._set_params = set_parameters_fn
        self._train_fn = train_fn
        self._evaluate_fn = evaluate_fn
        self._labels = list(label_iterable) if label_iterable is not None else None
        self._reporter = LabelDistributionReporter(
            num_classes=num_classes,
            class_names=class_names,
            normalise=True,
        )
        self._sim = straggler_simulator

    # ------------------------------------------------------------------ #
    # NumPyClient interface                                                 #
    # ------------------------------------------------------------------ #

    def get_parameters(self, config: Config) -> NDArrays:
        """Return current model parameters."""
        return self._get_params()

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Load parameters, train locally, and return updated parameters + metrics.

        If a :class:`StragglerSimulator` is attached and a drop event fires,
        this method raises :class:`RuntimeError` to simulate a client timeout
        from the server's perspective.

        Parameters
        ----------
        parameters:
            Global model parameters from the server.
        config:
            Round configuration dict.  FLIPS keys:
            - ``flips_report_label_dist``: bool — include label histogram.
            - ``proximal_mu`` (FedProx): float — proximal term weight.

        Returns
        -------
        (updated_parameters, num_samples, metrics)
        """
        if self._sim is not None:
            self._sim.maybe_drop()  # raises if drop event fires

        self._set_params(parameters)

        t0 = time.monotonic()
        if self._sim is not None:
            self._sim.maybe_delay()  # artificial sleep (test only)

        num_samples, fit_metrics = self._train_fn(config)
        train_time = time.monotonic() - t0

        # Always report timing and sample count
        fit_metrics[_TRAIN_TIME_KEY] = train_time
        fit_metrics[_NUM_SAMPLES_KEY] = num_samples

        # Report label distribution if server requested it
        if config.get(_LABEL_DIST_REQUEST_KEY, False) and self._labels is not None:
            ld_metrics = self._reporter.full_report(self._labels, prefix=_LABEL_DIST_PREFIX)
            fit_metrics.update(ld_metrics)

        return self._get_params(), num_samples, fit_metrics

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters on the local test set.

        Parameters
        ----------
        parameters:
            Global model parameters from the server.
        config:
            Evaluation configuration dict.

        Returns
        -------
        (loss, num_samples, metrics)
        """
        self._set_params(parameters)
        loss, num_samples, eval_metrics = self._evaluate_fn(config)
        return loss, num_samples, eval_metrics

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
"""FLIPS straggler simulation hook.

**Test-only** — not used in production federations.

Provides probability-based artificial delays and response-drops that can be
attached to :class:`~flwr.server.flips.client.flips_client.FlipsNumPyClient`
to simulate heterogeneous system conditions during experiments.
"""

from __future__ import annotations

import random
import time


class DropResponseError(RuntimeError):
    """Raised by :class:`StragglerSimulator` to simulate a dropped client response.

    The Flower server catches this as an exception in the ``failures`` list,
    which the FLIPS strategy interprets as a straggler.
    """


class StragglerSimulator:
    """Inject artificial latency or drop client responses for testing.

    All randomness is controlled by an explicit seed so experiment results
    are reproducible.

    Parameters
    ----------
    drop_probability:
        Probability in [0, 1) that :meth:`maybe_drop` raises
        :class:`DropResponseError` (simulating a client that never responds).
    delay_probability:
        Probability in [0, 1) that :meth:`maybe_delay` injects an artificial
        sleep.
    delay_seconds:
        Duration of the artificial sleep in seconds when triggered.
    seed:
        Random seed.  When ``None`` a fresh seed is generated automatically.
    """

    def __init__(
        self,
        drop_probability: float = 0.0,
        delay_probability: float = 0.0,
        delay_seconds: float = 2.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= drop_probability < 1.0:
            raise ValueError(
                f"drop_probability must be in [0, 1), got {drop_probability}"
            )
        if not 0.0 <= delay_probability < 1.0:
            raise ValueError(
                f"delay_probability must be in [0, 1), got {delay_probability}"
            )
        if delay_seconds < 0.0:
            raise ValueError(f"delay_seconds must be >= 0, got {delay_seconds}")

        self._drop_prob = drop_probability
        self._delay_prob = delay_probability
        self._delay_secs = delay_seconds
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def maybe_drop(self) -> None:
        """Possibly raise :class:`DropResponseError` to simulate a timeout.

        Call this at the **beginning** of
        :meth:`~flwr.server.flips.client.flips_client.FlipsNumPyClient.fit`
        before any training work is done.

        Raises
        ------
        DropResponseError
            With probability ``drop_probability``.
        """
        if self._drop_prob > 0.0 and self._rng.random() < self._drop_prob:
            raise DropResponseError("Simulated client drop (FLIPS straggler simulation)")

    def maybe_delay(self) -> None:
        """Possibly inject an artificial sleep to simulate a slow client.

        Call this **inside** the training loop after
        :meth:`maybe_drop` has been checked.
        """
        if self._delay_prob > 0.0 and self._rng.random() < self._delay_prob:
            time.sleep(self._delay_secs)

    @property
    def drop_probability(self) -> float:
        """The configured drop probability."""
        return self._drop_prob

    @property
    def delay_probability(self) -> float:
        """The configured delay probability."""
        return self._delay_prob

    @property
    def delay_seconds(self) -> float:
        """The configured delay duration in seconds."""
        return self._delay_secs

    def __repr__(self) -> str:
        return (
            f"StragglerSimulator("
            f"drop_p={self._drop_prob}, "
            f"delay_p={self._delay_prob}, "
            f"delay_s={self._delay_secs})"
        )

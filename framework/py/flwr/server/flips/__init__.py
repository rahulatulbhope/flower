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
"""FLIPS: Federated Learning using Intelligent Participant Selection.

Server-side modules:
- :mod:`flwr.server.flips.server.metadata_registry`  — per-client state store
- :mod:`flwr.server.flips.server.clustering`          — label-distribution k-means
- :mod:`flwr.server.flips.server.selector`            — cluster-aware round selection
- :mod:`flwr.server.flips.server.straggler`           — straggler tracking & overprovisioning
- :mod:`flwr.server.flips.server.instrumentation`     — round metrics & JSONL logging
- :mod:`flwr.server.flips.server.aggregation`         — FedAvg/FedProx/FedYogi adapters

Client-side modules:
- :mod:`flwr.server.flips.client.label_reporter`     — local label-histogram reporter
- :mod:`flwr.server.flips.client.flips_client`       — lightweight NumPyClient wrapper
- :mod:`flwr.server.flips.client.straggler_sim`      — test-only straggler simulation

Quick-start
-----------
::

    from flwr.server.flips.server.aggregation import make_flips_fedavg

    strategy = make_flips_fedavg(
        clients_per_round=10,
        min_fit_clients=8,
        min_available_clients=20,
    )
"""

from flwr.server.flips.server.aggregation import (
    make_flips_fedavg,
    make_flips_fedprox,
    make_flips_fedyogi,
    make_flips_strategy,
)
from flwr.server.flips.server.clustering import LabelDistributionClusterer
from flwr.server.flips.server.instrumentation import MetricsLogger, RoundMetrics
from flwr.server.flips.server.metadata_registry import ClientMetadata, MetadataRegistry
from flwr.server.flips.server.selector import FlipsSelector
from flwr.server.flips.server.straggler import StragglerTracker

__all__ = [
    "ClientMetadata",
    "FlipsSelector",
    "LabelDistributionClusterer",
    "MetadataRegistry",
    "MetricsLogger",
    "RoundMetrics",
    "StragglerTracker",
    "make_flips_fedavg",
    "make_flips_fedprox",
    "make_flips_fedyogi",
    "make_flips_strategy",
]

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
"""FLIPS server-side components."""

from flwr.server.flips.server.metadata_registry import ClientMetadata, MetadataRegistry
from flwr.server.flips.server.clustering import LabelDistributionClusterer
from flwr.server.flips.server.selector import FlipsSelector
from flwr.server.flips.server.straggler import StragglerTracker
from flwr.server.flips.server.instrumentation import RoundMetrics, MetricsLogger

__all__ = [
    "ClientMetadata",
    "MetadataRegistry",
    "LabelDistributionClusterer",
    "FlipsSelector",
    "StragglerTracker",
    "RoundMetrics",
    "MetricsLogger",
]

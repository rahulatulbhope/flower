# FLIPS: Federated Learning using Intelligent Participant Selection

FLIPS is a server-side Flower extension that improves client selection
fairness in non-IID federated learning settings by clustering clients
according to their label distributions and performing round-robin,
cluster-aware selection each round.

---

## Setup

Install the Flower fork in editable mode from its source tree:

```bash
cd /path/to/flower/framework/py
pip install -e .
```

NumPy is the only FLIPS-specific dependency; it is already required by Flower.

---

## Running with Flower

FLIPS plugs directly into Flower's **Strategy API**.  The recommended way to run
any Flower project is with the `flwr run` CLI, which handles both simulation
(single machine) and deployment (multi-machine) without touching application
code.

### 1 — Project layout

Create a new directory (e.g. `my_flips_run/`) with the following three files:

```
my_flips_run/
├── pyproject.toml
├── server_app.py
└── client_app.py
```

### 2 — `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-flips-run"
version = "0.1.0"
dependencies = ["flwr[simulation]>=1.13.0", "numpy"]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "you"

[tool.flwr.app.components]
serverapp = "server_app:app"
clientapp = "client_app:app"

[tool.flwr.app.config]
num-server-rounds = 10
clients-per-round = 5
num-supernodes = 20
num-classes = 10
```

### 3 — `server_app.py`

```python
from pathlib import Path

import numpy as np
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from flwr.server.flips.server.aggregation import make_flips_fedavg


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    num_rounds: int = cfg["num-server-rounds"]
    clients_per_round: int = cfg["clients-per-round"]
    num_classes: int = cfg["num-classes"]

    # Dummy initial parameters — replace with your real model weights
    initial_params = ndarrays_to_parameters(
        [np.zeros((num_classes, num_classes), dtype=np.float32)]
    )

    strategy = make_flips_fedavg(
        clients_per_round=clients_per_round,
        min_fit_clients=max(1, clients_per_round - 2),
        min_available_clients=clients_per_round,
        initial_parameters=initial_params,
        log_path=Path("flips_metrics.jsonl"),  # remove to disable logging
        # Optional: auto-select k clusters
        # clusterer=LabelDistributionClusterer(k=None, k_min=2, k_max=8),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
```

### 4 — `client_app.py`

```python
import time

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays

from flwr.server.flips.client.flips_client import FlipsNumPyClient


def client_fn(context: Context) -> NumPyClient:
    partition_id: int = context.node_config["partition-id"]
    num_partitions: int = context.node_config["num-partitions"]
    num_classes: int = int(context.run_config["num-classes"])

    # ------------------------------------------------------------------ #
    # Replace the stubs below with your real dataset / model loading.     #
    # ------------------------------------------------------------------ #
    # Simulate non-IID label skew: each partition gets 2 dominant classes
    dominant = [partition_id % num_classes, (partition_id + 1) % num_classes]
    rng = np.random.default_rng(partition_id)
    labels = rng.choice(dominant, size=500).tolist()

    # Toy model: single weight matrix
    weights = [np.zeros((num_classes, num_classes), dtype=np.float32)]

    def get_parameters() -> NDArrays:
        return weights

    def set_parameters(params: NDArrays) -> None:
        weights[:] = params

    def train(config: dict) -> tuple[NDArrays, int, dict]:
        # Replace with real training loop
        time.sleep(0.01)
        return weights, len(labels), {}

    def evaluate(config: dict) -> tuple[float, int, dict]:
        return 0.0, len(labels), {"accuracy": 0.0}

    return FlipsNumPyClient(
        get_parameters_fn=get_parameters,
        set_parameters_fn=set_parameters,
        train_fn=train,
        evaluate_fn=evaluate,
        label_iterable=labels,
        num_classes=num_classes,
    ).to_client()


app = ClientApp(client_fn=client_fn)
```

### 5 — Run in simulation mode

```bash
cd my_flips_run
flwr run .
```

Override any config value at the command line without editing files:

```bash
# Run for 20 rounds with 8 clients per round
flwr run . --run-config "num-server-rounds=20 clients-per-round=8"

# Scale up to 50 simulated nodes
flwr run . --run-config "num-supernodes=50 clients-per-round=10"
```

### 6 — Run in deployment mode

Start the SuperLink (server-side infrastructure):

```bash
flower-superlink --insecure
```

Start each SuperNode (one per physical client machine):

```bash
flower-supernode --insecure --superlink 127.0.0.1:9092
```

Then dispatch the run from your coordinator:

```bash
flwr run . --app . --stream
```

See the [Flower Deployment Engine docs](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
for TLS, authentication, and Docker setup.

---

## Inspecting FLIPS metrics

When `log_path` is set, FLIPS writes one JSON object per round to
`flips_metrics.jsonl`.  Read it with standard tools:

```python
import json, pathlib

records = [
    json.loads(line)
    for line in pathlib.Path("flips_metrics.jsonl").read_text().splitlines()
]

for r in records:
    print(
        f"Round {r['server_round']:3d} | "
        f"clients_selected={r['clients_selected']} | "
        f"straggler_rate={r['straggler_rate']:.2f} | "
        f"num_clusters={r['num_clusters']}"
    )
```

Available fields per record:

| Field | Type | Description |
|---|---|---|
| `server_round` | int | Round number |
| `clients_selected` | int | Clients sent FitIns |
| `clients_completed` | int | Clients that returned FitRes |
| `straggler_rate` | float | EMA straggler rate estimate |
| `num_clusters` | int | Active cluster count |
| `cluster_sizes` | list[int] | Members per cluster |
| `extra_clients` | int | Overprovisioned extras this round |
| `fit_duration_s` | float | Wall-clock time for this round's fit phase |

---

## Module layout

```
framework/py/flwr/server/flips/
├── __init__.py                        # Public API exports
├── server/
│   ├── metadata_registry.py           # Per-client state store
│   ├── clustering.py                  # Label-distribution k-means
│   ├── selector.py                    # Cluster-aware round selection
│   ├── straggler.py                   # Straggler tracking & overprovisioning
│   ├── instrumentation.py             # Round metrics & JSONL logging
│   └── aggregation.py                 # FedAvg / FedProx / FedYogi adapters
└── client/
    ├── label_reporter.py              # Local label-histogram reporter
    ├── flips_client.py                # Lightweight NumPyClient wrapper
    └── straggler_sim.py               # Test-only straggler simulation
```

---

## Architecture overview

### Server responsibilities

| Concern | Module | When |
|---|---|---|
| Register connected clients | `MetadataRegistry` | Every round (configure_fit) |
| Collect label distributions | `FlipsStrategyMixin` | Round 1 + every N rounds |
| Cluster clients | `LabelDistributionClusterer` | After receiving distributions |
| Select clients (cluster-aware) | `FlipsSelector` | Every round |
| Track stragglers | `StragglerTracker` | After aggregate_fit |
| Overprovision if needed | `FlipsSelector.select_overprovision` | When straggler rate > 0 |
| Aggregate gradients | Base strategy (FedAvg/FedProx/FedYogi) | Every round |
| Emit round metrics | `MetricsLogger` | Every round |

### Client responsibilities

| Concern | Module |
|---|---|
| Compute label histogram | `LabelDistributionReporter` |
| Local training (FedAvg / FedProx) | `FlipsNumPyClient.fit()` |
| Report timing metadata | Automatic in `fit()` |
| Evaluate local model | `FlipsNumPyClient.evaluate()` |
| (Test-only) Simulate stragglers | `StragglerSimulator` |

The client does **not** participate in clustering or selection.  It only reports
an aggregate label histogram and runtime metrics — no raw training data leaves
the client.

---

## Execution flow (per round)

```
Server                                         Clients
──────────────────────────────────────────────────────────
configure_fit(round R):
  1. Register any new ClientProxy IDs
  2. If round==1 or refresh due: set
     config["flips_report_label_dist"] = True
  3. If re-cluster due: run LabelDistributionClusterer
  4. Compute overprovision extra from StragglerTracker
  5. Call FlipsSelector.select() or .select_overprovision()
  6. Record selections in MetadataRegistry
  7. Return (selected_proxy, FitIns) pairs
                                               fit(parameters, config):
                                                 set_parameters(parameters)
                                                 [optional delay/drop sim]
                                                 run local training
                                                 if config["flips_report_label_dist"]:
                                                   compute label histogram
                                                   add to metrics["flips_ld_*"]
                                                 add metrics["flips_train_time_s"]
                                                 return updated params + metrics

aggregate_fit(round R, results, failures):
  1. Parse label distributions from FitRes.metrics
  2. Update MetadataRegistry label dists + straggler EMA
  3. Re-cluster if this was a label-collection round
  4. Record straggler outcomes in StragglerTracker
  5. Delegate to parent.aggregate_fit() (FedAvg/FedProx/FedYogi)
  6. Call evaluate() for server-side test metrics
  7. Emit RoundMetrics via MetricsLogger
  8. Return (aggregated_parameters, fit_metrics)
```

---

## API quick-start

The three factory functions cover the most common base strategies.  All
keyword arguments not listed here are forwarded unchanged to the underlying
strategy constructor.

### FedAvg

```python
from flwr.server.flips.server.aggregation import make_flips_fedavg

strategy = make_flips_fedavg(
    clients_per_round=10,
    min_fit_clients=8,
    min_available_clients=20,
    initial_parameters=...,
    log_path=Path("logs/flips.jsonl"),
)
```

### FedProx with straggler overprovisioning

```python
from flwr.server.flips.server.aggregation import make_flips_fedprox
from flwr.server.flips.server.straggler import StragglerTracker

tracker = StragglerTracker(window=5, min_overprovision=2)

strategy = make_flips_fedprox(
    clients_per_round=10,
    proximal_mu=0.01,
    min_fit_clients=8,
    min_available_clients=20,
    straggler_tracker=tracker,
    initial_parameters=...,
)
```

### Auto-cluster-k selection

```python
from flwr.server.flips.server.clustering import LabelDistributionClusterer

clusterer = LabelDistributionClusterer(k=None, k_min=2, k_max=8, seed=42)
strategy = make_flips_fedavg(
    clients_per_round=10,
    clusterer=clusterer,
    initial_parameters=...,
)
```

### Client setup

```python
from flwr.server.flips.client.flips_client import FlipsNumPyClient
from flwr.client import ClientApp

def client_fn(context):
    client = FlipsNumPyClient(
        get_parameters_fn=lambda: get_weights(net),
        set_parameters_fn=lambda p: set_weights(net, p),
        train_fn=lambda cfg: train(net, trainloader, cfg),
        evaluate_fn=lambda cfg: test(net, testloader),
        label_iterable=train_dataset.targets,
        num_classes=10,
    )
    return client.to_client()

app = ClientApp(client_fn=client_fn)
```

---

## Where FLIPS modifies standard Flower behaviour

FLIPS replaces the `configure_fit` and `aggregate_fit` methods of the chosen
base strategy.

| Standard Flower behaviour | FLIPS change |
|---|---|
| `configure_fit` randomly samples `fraction_fit * available` clients | Cluster-aware quota-based selection via `FlipsSelector` |
| No client-side metadata tracked server-side | `MetadataRegistry` maintains per-client label dist, cluster ID, pick count, straggler probability |
| No cluster concept | `LabelDistributionClusterer` groups clients; assignments stored in registry |
| No straggler compensation | `StragglerTracker` estimates rate; selector overprovisioned accordingly |
| `aggregate_fit` passes results straight to parent | FLIPS side-effects (registry updates, straggler recording, metric emission) run first |

---

## Minimal patch plan for this Flower fork

The FLIPS extension lives entirely in `framework/py/flwr/server/flips/` and
requires **no modifications** to any existing Flower file.  The table below
lists the vanilla Flower files you would need to touch only if you want to
expose FLIPS as a first-class Flower namespace or add deep workflow integration.

| File | Change needed | Required? |
|---|---|---|
| `framework/py/flwr/server/__init__.py` | `from .flips import ...` | Optional — for `flwr.server.flips.*` shorthand |
| `framework/py/flwr/server/strategy/__init__.py` | Export `make_flips_fedavg` etc. | Optional |
| `framework/py/flwr/server/workflow/default_workflows.py` | Integrate FLIPS metrics logger into the default workflow loop | Only if you want FLIPS metrics in the built-in workflow telemetry |
| `framework/py/flwr/server/server.py` | Pass straggler cutoff timeout to `fit` calls | Only if you want server-enforced round timeouts to be tracked by `StragglerTracker` |
| `framework/pyproject.toml` | Add `flips` as an optional dependency group listing `numpy` | Recommended |

**Nothing in `flwr/client/` needs to change.**  The `FlipsNumPyClient` subclasses
`NumPyClient` from the standard Flower client path.

---

## Running the tests

```bash
# From the repo root — requires Python >= 3.10
cd /path/to/flower
python3.11 -m pytest framework/py/flwr/server/flips/tests/ -v

# Or from the framework/py/ directory after pip install -e .
cd framework/py
python3.11 -m pytest flwr/server/flips/tests/ -v
```

Individual test files and what they cover:

| File | Coverage |
|---|---|
| `test_label_distribution_reporting.py` | `LabelDistributionReporter` — histograms, edge cases |
| `test_clustering.py` | `_kmeans`, `_davies_bouldin`, `LabelDistributionClusterer` |
| `test_selector.py` | `FlipsSelector` — fairness, pick-count preference, overprovisioning |
| `test_straggler_logic.py` | `StragglerTracker` — rate estimation, EMA, overprovisioning count |
| `test_aggregation_compat.py` | `make_flips_fedavg/fedprox/fedyogi` — mock round lifecycle |
| `test_end_to_end_simulation.py` | Multi-round non-IID simulation, metric logging, JSONL output |

---

## Key design decisions

1. **Server-only clustering.** Label distributions are collected as aggregate
   histograms in `FitRes.metrics` — no raw data leaves the client.

2. **Modular aggregation.** `FlipsStrategyMixin` is composed onto any
   FedAvg-derived strategy via `make_flips_strategy()`.  The mixin only
   overrides `configure_fit` and `aggregate_fit`; all aggregation maths are
   handled by the base class unchanged.

3. **Pure-Python k-means.** `clustering.py` uses only NumPy — no scikit-learn
   dependency — so it can run inside the Flower server without introducing an
   additional heavy dependency.  Davies-Bouldin index is used for auto-k
   because it is fast and does not require a ground-truth holdout.

4. **Deterministic selection.** `FlipsSelector` sorts by `(pick_count, cid)`
   which gives fully deterministic output for a given registry state, enabling
   reproducible experiments without a global RNG.

5. **Straggler simulation is test-only.** `StragglerSimulator` lives in
   `client/straggler_sim.py` and is only attached to `FlipsNumPyClient`
   during experiments.  The production code path never imports or calls it.

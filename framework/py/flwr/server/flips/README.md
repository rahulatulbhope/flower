# FLIPS: Federated Learning using Intelligent Participant Selection

FLIPS is a server-side Flower extension that improves client selection
fairness in non-IID federated learning settings by clustering clients
according to their label distributions and performing round-robin,
cluster-aware selection each round.

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

## Quick-start

### Minimal FedAvg setup

```python
from flwr.server.flips.server.aggregation import make_flips_fedavg
from flwr.server.server_config import ServerConfig
from flwr.server.server_app import ServerApp

strategy = make_flips_fedavg(
    clients_per_round=10,
    min_fit_clients=8,
    min_available_clients=20,
    initial_parameters=...,   # your model parameters
    log_path=Path("logs/flips.jsonl"),
)

app = ServerApp(
    server_fn=lambda ctx: ServerAppComponents(
        strategy=strategy,
        server_config=ServerConfig(num_rounds=50),
    )
)
```

### FedProx with automatic straggler overprovisioning

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
    ...
)
```

### Client setup

```python
from flwr.server.flips.client.flips_client import FlipsNumPyClient
from flwr.client import ClientApp, NumPyClient

client = FlipsNumPyClient(
    get_parameters_fn=lambda: get_weights(net),
    set_parameters_fn=lambda p: set_weights(net, p),
    train_fn=lambda cfg: train(net, trainloader, cfg),
    evaluate_fn=lambda cfg: test(net, testloader),
    label_iterable=train_dataset.targets,  # any iterable of labels
)
app = ClientApp(client_fn=lambda ctx: client)
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
# Requires Python >= 3.10 and numpy installed
python3.11 -m pytest framework/py/flwr/server/flips/tests/ -v
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

# Elementary-RAM Dijkstra

The `graph-ram` task expresses priority-queue Dijkstra over a generic word-addressed register machine. The environment exposes exactly eight actions:

```text
MOV LOAD STORE ADD SUB SHL1 SHR1 CMP
```

Graph records and scratch storage are ordinary memory words. The learned traces implement array addressing, parent and child arithmetic, record swaps, insertion, extraction, stale-entry handling, edge traversal, and relaxation. The environment has no heap object, priority operation, minimum operation, or Dijkstra operation.

The model chooses only fixed opcodes and registers. Addresses, node references, distances, weights, indices, and memory contents remain environment-owned values and are never classification targets.

The learned hierarchy is:

```text
DIJKSTRA
  INITIALIZE
  INSERT
    BUBBLE_UP
  EXTRACT
    BUBBLE_DOWN
  SCAN_EDGES
    RELAX
      INSERT
  ACT
```

For a binary priority queue, execution is `O((V + E) log V)`. On the connected Erdos-Renyi training distribution, `E = Theta(V log V)`, yielding approximately `O(V log^2 V)`.

Train or resume the task with:

```bash
uv run python main.py graph-ram \
  --maximum-train-nodes 30 \
  --training-examples-per-size 4 \
  --steps 500000 \
  --weight-video artifacts/graph_ram_dijkstra/weight_evolution.mp4 \
  --video-frame-interval 50 \
  --log-interval 1000
```

The elementary ISA increases the number of neural decisions per algorithmic operation. It removes data-structure prior knowledge from actions, but has a larger constant factor than the higher-level graph pointer machine.

## Regularization And Size

Use explicit L2 loss regularization without L1 or weight decay:

```bash
uv run python main.py graph-ram \
  --weight-decay 0 \
  --l1-regularization 0 \
  --l2-regularization 1e-11
```

Model capacity is controlled independently:

```text
--state-size
--program-size
--key-size
--hidden-size
--layers
```

Every checkpoint records these dimensions and the exact trainable parameter count. L1, L2, and decoupled weight decay are mutually exclusive trainer modes.

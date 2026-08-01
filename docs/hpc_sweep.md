# Multi-GPU Generalization Sweep

The graph sweep schedules the Cartesian product of maximum training sizes and seeds through a shared multiprocessing queue. Every worker selects exactly one physical GPU before TensorFlow initializes it.

## Capacity Definition

For each checkpoint, candidate node counts are tested in ascending order. A candidate passes only when all 100 randomized problems have:

- exact NetworkX shortest-path distances;
- valid shortest-path parent trees;
- no invalid learned action;
- no recursion or step-limit failure.

The first failed candidate stops the search. The reported capacity is therefore a conservative contiguous bound.

## Command

```bash
uv run python main.py graph-sweep \
  --gpu-count 8 \
  --maximum-train-nodes 5,10,15,20 \
  --seeds 1,2 \
  --checkpoint-steps 1000,2000,4000,8000,12000,20000,30000,40000,50000,60000 \
  --generalization-nodes 10,20,30,40,50,75,100,125,150,200
```

Provide at least as many `(training maximum, seed)` combinations as GPUs for full utilization.

## Output

```text
configuration.json
train_max_XXXX/seed_XXXX/checkpoints/*.weights.h5
train_max_XXXX/seed_XXXX/evaluations/*.json
train_max_XXXX/seed_XXXX/resume/
sweep_results.json
sweep_results.csv
generalization_vs_steps.png
generalization_vs_steps.pdf
```

Use a new output directory for a different protocol. Delete the output directory to start from scratch. TensorFlow optimizer and model state are resumable from the most recent sampled step.

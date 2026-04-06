# From Problem to Code

This page walks you through your first estimator — from understanding the problem to seeing a real score.

## The challenge in 30 seconds

You receive a random MLP (multilayer perceptron) with ReLU activations. Your job: predict the expected value of each neuron after each layer, without running thousands of forward passes.

The MLP has a fixed `width` (neurons per layer) and `depth` (number of layers). Each layer applies a weight matrix and a ReLU activation: `output = max(0, W @ input)`. The input to the network is standard normal noise — one independent N(0,1) value per neuron.

Your estimator receives the MLP's weight matrices and a FLOP budget. It returns a `(depth, width)` array of predicted neuron means. The closer to ground truth, the better your score.

## A concrete example

Consider a tiny MLP with width=4 and depth=2:

- **Input:** 4 random values drawn from N(0,1)
- **Layer 1:** multiply by a 4x4 weight matrix, apply ReLU
- **Layer 2:** multiply by another 4x4 weight matrix, apply ReLU

Ground truth (from averaging 10,000 forward passes) might look like:

| | Neuron 0 | Neuron 1 | Neuron 2 | Neuron 3 |
|---|---|---|---|---|
| Layer 1 | 0.42 | 0.38 | 0.51 | 0.29 |
| Layer 2 | 0.15 | 0.22 | 0.18 | 0.11 |

Your estimator must predict these values — without running 10,000 forward passes.

## Run the zeros baseline

Start with the template estimator (it returns all zeros):

```bash
nestim init ./my-estimator
nestim run --estimator ./my-estimator/estimator.py --n-mlps 3
```

You will see a score report. Look for `primary_score` — this is the MSE of your final-layer predictions vs ground truth. With all-zeros predictions, it will be nonzero (you are wrong about everything).

## Your first real estimator

Copy the mean propagation example — it uses the ReLU expectation formula to predict neuron means analytically:

```bash
cp examples/estimators/mean_propagation.py ./my-estimator/estimator.py
nestim run --estimator ./my-estimator/estimator.py --n-mlps 3
```

Compare the `primary_score` to the zeros baseline. It should be significantly lower — mean propagation uses the network's weight matrices to make informed predictions instead of guessing.

**Why it works:** For a Gaussian input passing through a ReLU, the expected output has a closed-form expression involving the normal CDF and PDF. Mean propagation chains this formula through each layer, tracking per-neuron means and variances. See [Algorithm Ideas](../how-to/algorithm-ideas.md) for details.

## Reading the score report

The per-MLP section of the report shows:

- **`final_mse`**: MSE of your last-layer predictions vs ground truth. This is your primary score. Lower is better.
- **`all_layer_mse`**: MSE across all layers (secondary score).
- **`flops_used`**: how many FLOPs your estimator consumed for this MLP.
- **`budget_exhausted`**: `true` if you exceeded the FLOP budget — your predictions were zeroed.

If `budget_exhausted` is `true`, your estimator is too expensive. See [Manage Your FLOP Budget](../how-to/manage-flop-budget.md).

## Improve from here

You now have a working estimator. Next steps:

1. **Try the combined estimator** — it routes between cheap and expensive algorithms based on budget:
   ```bash
   cp examples/estimators/combined_estimator.py ./my-estimator/estimator.py
   nestim run --estimator ./my-estimator/estimator.py --n-mlps 3
   ```

2. **Understand the algorithms** — [Algorithm Ideas](../how-to/algorithm-ideas.md) surveys estimation strategies with FLOP costs and tradeoffs.

3. **Optimize your budget** — [Manage Your FLOP Budget](../how-to/manage-flop-budget.md) shows how to check costs and stay within budget.

4. **Debug when stuck** — [Debugging Checklist](../how-to/debugging-checklist.md) gives a systematic diagnosis flow.

## Next step

- [Algorithm Ideas](../how-to/algorithm-ideas.md)
- [Write an Estimator](../how-to/write-an-estimator.md)
- [Scoring Model](../concepts/scoring-model.md)

# CRANE research plan

How to turn [THEORY.md](THEORY.md) into results that mean something. Written before any experiments, so the questions and success criteria are fixed in advance.

## Hypotheses

The question is not "does memory help" (on tasks built to need memory, almost any memory helps). The question is whether **this** memory, a graph of events and consequences with value-weighted links, beats a simple one.

| # | Hypothesis | Compared against |
|---|---|---|
| H1 | Memory helps on synthetic worlds (THEORY §7) | the same model with memory off |
| H2 | An event graph with typed links beats a flat vector store | plain kNN memory over the same written vectors |
| H3 | Valence-weighted link updates help on tasks with a reward | the same graph with unweighted Hebbian updates |
| H4 | Consolidation shrinks memory without losing quality | the same graph without consolidation |

H2 is the real research question. The kNN baseline is mandatory; without it a win proves nothing.

## Setup

- Small scale first: character-level tokenizer, a small thinking network, synthetic episodes. One run should take 10–60 minutes on an RTX 4070 (12 GB).
- Tasks: "predict what happens next" in a situation, and recalling facts introduced earlier in an episode. Each episode has its own objects, rules and facts, so memorising across episodes is useless.
- A fixed held-out test set that is never used for tuning. At least 3 seeds per configuration. Report mean and spread.
- Metrics fixed in advance: task accuracy / loss on the test set, memory size over time, retrieval hit rate (did the needed memory come back).

## Phases

1. **Literature check.** Where has this been done (RETRO, kNN-LM, Memorizing Transformers, Product Key Memory, Titans, NTM/DNC, ATOMIC/COMET, latent reasoning work)? What is actually new here? Output: a short positioning note.
2. **Fix the scaffold** against THEORY §9: meaningful memory units, aligned key-value retrieval, trained confidence, wider thought bottleneck, train/inference consistency.
3. **Synthetic world generator + baselines:** memory off, kNN memory.
4. **Event graph:** typed links, memory regions, then valence, then consolidation, each as its own ablation row.
5. **Write-up:** table, plots, what worked, what did not and why. Negative results are reported too.

## Rules for running it with agents

- Planning and result analysis on the strongest model (Fable); code on Sonnet; mechanical checks on Haiku. Use the setup from github.com/nezo32/claude-code-setup and work from a session opened in this repository.
- Agents never declare a hypothesis confirmed. They report numbers; the owner reviews after every phase before the next one starts.
- No tuning on the test set, no changing metrics after seeing results, no dropping seeds that look bad.
- Every experiment is reproducible from a config file and a seed; results go to `results/` with the config that produced them.
- Watch for false wins: leakage of test facts into training, weak baselines, a "win" that disappears with another seed.

## Expectations

- Very likely: a scaffold that matches the theory, a synthetic benchmark, honest baselines and an ablation table.
- Plausible: memory clearly beats no memory (H1).
- Open: whether the event graph beats plain kNN memory (H2). That is the point of the project; either answer is worth publishing if the experiment is clean.

Time-boxed side project: weekends, not a full-time effort.

# CRANE — Cognitive Retrieval Augmented Neural Engine

A research skeleton for a non-autoregressive, memory-centric language model. Instead of next-token prediction, CRANE iteratively forms a continuous *thought* by associating the input with a weighted graph of memory units, and only then projects that thought back into language.

## Principles

1. Thought is not language
2. Memory is the center of cognition
3. The learning is process of updating memory-units connections as well as updating processing contour
4. Reasoning emerges from associative activation
5. Experience shapes memory topology
6. Language is an interface to cognition
7. Prediction error updates memory
8. Attention controls activation dynamics

## Architecture

```
text
  -> Tokenizer
  -> Embedder
  -> SalientExtractor              (K learnable slots, attention-pool)
  -> MemoryGraphStore.retrieve     (vector search + 1..hops graph expansion)
  -> MemorySignificanceFilter      (sim + edge-weight + activation)
  -> MemoryFusion                  (significance-biased cross-attention)
  -> Reasoner                      (transformer encoder + state token)
  -> ThoughtHead                   (thought_vector, confidence)
        |-- confidence < threshold -> write back + reinforce + decay -> loop
        +-- confidence >= threshold -> ThoughtLanguageDecoder
                                            -> Tokenizer.decode
                                                   -> text
```

Every iteration:

- writes the new thought into the memory graph with weighted links to the units it consumed,
- Hebbian-reinforces the edges between all co-activated nodes,
- applies a global multiplicative decay so unused links fade.

This realizes the README principles: memory topology is updated by usage, association weights *are* the learning signal, and language is only the interface, not the medium of cognition.

## Project layout

```
crane/
  language/
    tokenization/{base,modules}          # Tokenizer ABC + CharTokenizer
    embeddings/{base,modules}            # Embedder ABC + LookupEmbedder
  neural/
    salient/{base,modules}               # SalientExtractor + AttentionPoolSalientExtractor
    fusion/{base,modules}                # MemoryFusion + CrossAttentionMemoryFusion
    reasoning/{base,modules}             # Reasoner + TransformerReasoner
    thought/{base,modules}               # ThoughtHead + LinearThoughtHead
    decoder/{base,modules}               # ThoughtLanguageDecoder + TransformerThoughtDecoder
  memory/
    graph/
      base/                              # MemoryNode / MemoryEdge / RetrievalResult
                                         # MemoryGraphStore (ABC), MemorySignificanceFilter (ABC)
      modules/                           # Neo4jMemoryGraphStore + ActivationSignificance
  cognition/
    engine/{base,modules}                # CognitiveEngine (ABC) + IterativeCognitiveEngine
cli/
  scripts/generate_inits.py              # regenerates auto-export __init__.py blocks
pipelines/
  main.py                                # demo: wires everything with random weights
docker-compose.yml                       # Neo4j 5.x with vector index support
```

Convention: every domain has a `base/` package (abstract `ABC` contracts) and a `modules/` package (concrete implementations). New code is added under `modules/`, then `uv run generate-init` regenerates the `# <auto-exports>` blocks.

## Requirements

- Python `>=3.13` (managed by [`uv`](https://docs.astral.sh/uv/) — see [`.python-version`](.python-version))
- `torch>=2.11` (CUDA 13.0 wheel index is configured in [`pyproject.toml`](pyproject.toml); a CPU install works fine for the demo)
- Docker (for the Neo4j memory backend)

## Quick start

```bash
uv sync

docker compose up -d
# Neo4j Browser: http://localhost:7474
# Bolt:          bolt://localhost:7687
# Default creds: neo4j / craneneo4j  (overridable via NEO4J_USER / NEO4J_PASSWORD)

uv run python pipelines/main.py
```

The demo builds every component with `D=64`, 4 heads, 2 layers, initialises Neo4j with a cosine vector index on `MemoryUnit.vector`, and runs `engine.think("hello")` with random weights — useful for verifying the wiring before training.

### Environment variables

| Variable           | Default                  | Used by                    |
| ------------------ | ------------------------ | -------------------------- |
| `NEO4J_URI`        | `bolt://localhost:7687`  | `pipelines/main.py`        |
| `NEO4J_USER`       | `neo4j`                  | `pipelines/main.py` + compose |
| `NEO4J_PASSWORD`   | `craneneo4j`             | `pipelines/main.py` + compose |

## Memory schema (Neo4j 5.x)

```
(:MemoryUnit {
   id: string,
   vector: list<float>,        # indexed by `memory_vectors` (cosine)
   payload: string (JSON),
   created_at: int,
   activation: float
 })

-[:ASSOCIATED_WITH {
   weight: float,
   last_used: int
 }]->

(:MemoryUnit { ... })
```

The vector index and uniqueness constraint on `id` are created idempotently on first connect by `Neo4jMemoryGraphStore._ensure_schema`.

## Development

```bash
uv sync --group dev
uv run generate-init   # regenerate __init__.py auto-exports after adding files
uv run pytest          # (test suite not yet populated)
```

To add a new concrete component:

1. Put it under the appropriate `modules/` folder.
2. Subclass the matching `base/` ABC and (where relevant) `torch.nn.Module`.
3. Re-run `uv run generate-init`.

## Status

Skeleton stage — abstract contracts, a Neo4j driver, and reference `torch.nn` skeletons for every stage. There is no training loop, no losses, and no pretrained weights yet. The forward path is end-to-end runnable so the architecture can be exercised, profiled, and extended.

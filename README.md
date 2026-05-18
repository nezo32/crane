# CRANE — Cognitive Retrieval Augmented Neural Engine

Research codebase for a **memory-centric cognitive stack**: recurrent *thought formation* grounded in Neo4j (vector similarity + weighted graph traversal), fused with transformers, decoded to tokens. Built for **research and iteration**, not a production LLM replacement.

Alongside iterative inference (`think()` loop), there is a **supervised training path** (`pipelines/train.py`): next-token loss on GPT-2 BPE windows, gradients through the neural contour, heuristic **prediction-error feedback** onto graph edges (`engine.feedback`), and serialized **full checkpoints** (weights + optimizer + Neo4j snapshot).

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
  -> SalientExtractor              (learnable slots, attention pool)
  -> MemoryGraphStore.retrieve     (vector index + optional hop expansion)
  -> MemorySignificanceFilter       (similarity + edge weight + activation)
  -> MemoryFusion                   (bias over cross-attention from significance)
  -> Reasoner                       (TransformerEncoder + pooled state token)
  -> ThoughtHead                    (thought vector + confidence scalar)
        |-- confidence < threshold -> write-back + reinforce + decay (+ periodic prune)
        |                          -> loop (multi-step inference; see pipelines/main.py)
        +-- confidence >= threshold -> ThoughtLanguageDecoder -> Tokenizer.decode -> text
```

**Inference vs training**

- **`pipelines/main.py`**: builds the full stack with **`max_iterations=4`** cognitive loop → `think(prompt)`.
- **`pipelines/train.py`**: **`max_iterations=1`** — one salient→retrieve→fuse→reason→thought forward per minibatch keeps BPTT small. **`decay_factor=0.9995`**, **`prune_every=500`** (versus **`main.py`** `max_iterations=4`, **`0.999`**, **`prune_every=4`**).
- Decoder training is **teacher-forced** CE on GPT-2 BPE token slices; **`engine.feedback(score)`** nudges co-activated memory edges from loss-derived reward.

Every cognitive iteration **may** persist a qualifying thought (`write_confidence_min`), **reinforces** pairwise links among retrieved units, applies **activation bump** + **global decay**, and optionally **hard-prunes** weak edges cold nodes (`prune_every`).

## Reference model sizes (`main.py` / `train.py`)

| Setting | Value |
| -------- | ------- |
| Embedding / memory vector dim `D` | 256 |
| Attention heads | 8 |
| Salient slots | 16 |
| Reasoner Transformer layers | 4 |
| Decoder Transformer layers | 4 |
| Decoder max length (`generate`) | 128 |
| tokenizer | GPT-2 BPE (~50 257 vocab) via `BPETokenizer.from_pretrained("gpt2")` |
| corpus (training) default | `data/corpus.txt` (see `pipelines/fetch_corpus.py`) |

[`docker-compose.yml`](docker-compose.yml) only runs **Neo4j** — the CRANE trainer is local (`uv` / GPU optional).

## Project layout

```
crane/
  config/                               # optional .env (load_dotenv_if_present)
  language/
    tokenization/{base,modules}         # Tokenizer ABC, CharTokenizer, BPETokenizer (GPT-2)
    embeddings/{base,modules}           # Embedder ABC, LookupEmbedder
  neural/
    salient/, fusion/, reasoning/, thought/, decoder/
  memory/
    graph/base/                         # MemoryGraphStore (+ export/import), types, significance
    graph/modules/                      # Neo4jMemoryGraphStore, ActivationSignificance
  cognition/engine/                     # EngineConfig + IterativeCognitiveEngine
cli/scripts/generate_inits.py           # regenerate # <auto-exports> blocks
pipelines/
  main.py                               # inference demo — CUDA auto; GPT-2 BPE
  train.py                              # training + checkpoints + Neo4j snapshot
  checkpoint.py                         # prepare_training_startup / save_checkpoint
  fetch_corpus.py                       # download Tiny Shakespeare → data/corpus.txt
  wipe_memory.py                        # CRANE wipe (--all nukes database)
docker-compose.yml                      # Neo4j 5.x
.dockerignore                           # excludes .env* from docker build context
```

Convention: add concrete code under **`modules/`**, subclass matching **`base/`** ABCs (and `torch.nn.Module` where needed), then **`uv run generate-init`**.

## Requirements

| | |
| --- | --- |
| Python | `>=3.13` ([`.python-version`](.python-version); use [`uv`](https://docs.astral.sh/uv/) `sync`) |
| PyTorch | `>=2.11` optional CUDA wheels via pinned index in [`pyproject.toml`](pyproject.toml); CPU OK |
| Docker | Neo4j 5.x (vector index capable) |

Runtime libraries (declared): `torch`, `torchvision`, `neo4j`, **`tokenizers`**, **`python-dotenv`**, `pyyaml`. First GPT-2 tokenizer fetch uses the Hugging Face Hub (offline cache afterward).

## Quick start

```bash
uv sync

# optional secrets / overrides (Compose still wins over .env for keys already set):
cp .env.example .env

docker compose up -d
# Browser: http://localhost:7474   Bolt: bolt://localhost:7687
# default neo4j / craneneo4j unless overridden via env / compose

# optional training corpus (~1 MB plaintext)
uv run python pipelines/fetch_corpus.py

uv run python pipelines/train.py       # wipes Neo4j each run → may resume from checkpoints/latest.pt
uv run python pipelines/main.py       # iterative think("hello"); random module init unless you load weights
```

Run training and demo from the **repository root** so `pipelines/train.py` can `import checkpoint` beside it.

### Training prelude (every run of `pipelines/train.py`)

[`prepare_training_startup`](pipelines/checkpoint.py) enforces:

1. **Neo4j wipe** — clear `:MemoryUnit` graph, rebuild id constraint + cosine vector index.
2. **If** `CRANE_CHECKPOINT_PATH` exists — load **`model`** state dicts, **`optimizer`**, **`global_step`**, validate **`embedding_dim`** **and optional `tokenizer_name` vs `gpt2`**.
3. **`import_snapshot`** — restore **`neo4j_snapshot`** edges/nodes into the emptied DB.

Then training runs **`TRAIN_STEPS`** iterations; checkpoint written every **`CRANE_SAVE_EVERY`** **`training_step`** ticks and once at exit.

Checkpoint payload: **`checkpoint_version`**, **`embedding_dim`**, **`tokenizer_name`**, **`global_step`**, **`model`**, **`optimizer`** (nested tensors moved CPU for save), **`neo4j_snapshot`** (`nodes` / `edges`).

**Note:** `pipelines/main.py` does **not** yet **`torch.load`** a checkpoint — inference uses freshly initialized weights unless you extend it.

### Inference memory wipe CLI

```bash
uv run python pipelines/wipe_memory.py        # Neo4jMemoryGraphStore.wipe(dim=CRANE_EMBEDDING_DIM); recreates schema
uv run python pipelines/wipe_memory.py --all # delete every node label in DB (destructive)
```

## Environment variables

| Variable | Default | Role |
| -------- | ------- | ----- |
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt endpoint |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | `craneneo4j` | |
| `HF_TOKEN` | — | Higher HF Hub rate limits (optional); see [.env.example](.env.example) |
| `CRANE_CORPUS_URL` | Tiny Shakespeare URL | `fetch_corpus.py` |
| `CRANE_CORPUS_PATH` | `data/corpus.txt` | `fetch_corpus.py` |
| `CRANE_CHECKPOINT_PATH` | resolved `checkpoints/latest.pt` | train load/save |
| `CRANE_SAVE_EVERY` | `500` | periodic `.pt` while training |
| `CRANE_EMBEDDING_DIM` | `256` | must match **`D`** in code + `wipe_memory.py` Neo4j index |

### Local `.env`

[`crane/config/load_env.py`](crane/config/load_env.py) discovers **`.env`** from the cwd (walk upward) with **`override=False`**: injected production / Docker **`os.environ`** always wins.[`.dockerignore`](.dockerignore) keeps **`.env` / `.env.example`** out of image build contexts.

## Memory schema (Neo4j)

```
(:MemoryUnit {
   id: string,
   vector: list<float>,       # VECTOR INDEX memory_vectors cosine
   payload: string (JSON),
   created_at: int,
   activation: float
})

[:ASSOCIATED_WITH { weight: float, last_used: int }]
```

[`Neo4jMemoryGraphStore`](crane/memory/graph/modules/neo4j_store.py) **idempotently** runs **`_ensure_schema`**, **`wipe()`**, and **`export_snapshot` / `import_snapshot`** for checkpoints.

## Development

```bash
uv sync --group dev
uv run generate-init
uv run pytest    # scaffold only
```

Adding a neural module:

1. place under **`crane/*/modules/`**,
2. subclass the **`base/`** contract + `torch.nn.Module` where needed,
3. register it inside **`IterativeCognitiveEngine._TRAINABLE_MODULE_ATTRS`** and checkpoint dict keys (`module_state_dict` / `load_module_state_dict`) if it must persist.

## Status

Functional **research scaffold**: iterative inference demo, Neo4j-backed associative memory **with pruning / decay / Hebb-style reinforce + `feedback()`**, GPT-2 BPE, **CUDA auto device** in demos/training, corpus fetch helper, **`prepare_training_startup` wipe → restore → train**, periodic **checkpoint + Neo4j graph snapshot** resume. Inference **`main`** still uses **random-init weights** unless you load a checkpoint manually — wiring `load_checkpoint` there is natural next step.

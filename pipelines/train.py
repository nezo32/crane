"""
Next-token prediction training loop for CRANE.

Tokenizes the entire corpus once with byte-level BPE, then samples random
windows from that long token stream:

    prefix_ids = stream[start : start + window]
    target_ids = [BOS] + stream[start + window : start + window + window]

For each step the cognitive stack produces a single thought from `prefix_ids`,
the decoder is teacher-forced on `target_ids`, and we compute cross entropy.
`engine.feedback(score)` then translates the loss into a memory-side signal
(positive when low loss, negative when high) — the README's
"prediction error updates memory" principle made executable.

The cognitive loop is unrolled to ONE iteration during training; multi-step
"thinking" is reserved for inference (see `pipelines/main.py`).

Usage:
    docker compose up -d
    uv run python pipelines/fetch_corpus.py
    uv run python pipelines/train.py

Each run calls ``prepare_training_startup`` (see ``pipelines/checkpoint.py``):

1. **Wipe** Neo4j ``MemoryUnit`` graph and rebuild constraint + vector index.
2. **Load** weights + optimizer + ``global_step`` from ``CRANE_CHECKPOINT_PATH``
   (default ``checkpoints/latest.pt``) if that file exists.
3. **Import** the serialized Neo4j snapshot from the same file (if present).
4. **Train** for ``TRAIN_STEPS`` more optimizer steps, saving the checkpoint
   every ``CRANE_SAVE_EVERY`` steps and once at the end.
"""

import math
import os
import random
import time
from pathlib import Path

import torch

from crane.config import load_dotenv_if_present
from crane.cognition.engine.base.engine import EngineConfig
from crane.cognition.engine.modules.iterative_engine import IterativeCognitiveEngine
from crane.language.embeddings.modules.lookup_embedder import LookupEmbedder
from crane.language.tokenization.modules.bpe_tokenizer import BPETokenizer
from crane.memory.graph.modules.activation_significance import ActivationSignificance
from crane.memory.graph.modules.neo4j_store import Neo4jMemoryGraphStore
from crane.neural.decoder.modules.transformer_thought_decoder import (
    TransformerThoughtDecoder,
)
from crane.neural.fusion.modules.cross_attention_fusion import (
    CrossAttentionMemoryFusion,
)
from crane.neural.reasoning.modules.transformer_reasoner import TransformerReasoner
from crane.neural.salient.modules.attention_pool_extractor import (
    AttentionPoolSalientExtractor,
)
from crane.neural.thought.modules.linear_thought_head import LinearThoughtHead

from checkpoint import prepare_training_startup, save_checkpoint


load_dotenv_if_present()


# Model dims — RTX 4070 12 GB can comfortably handle these.
D = 256
HEADS = 8
SALIENT_SLOTS = 16
REASONER_LAYERS = 4
DECODER_LAYERS = 4
DECODER_MAX_LEN = 128

# Tokenizer — uses GPT-2's ready-trained byte-level BPE (50,257 tokens).
# Downloaded once via HuggingFace Hub, cached in ~/.cache/huggingface.
TOKENIZER_NAME = "gpt2"

# Training
WINDOW_TOKENS = 64          # prefix length AND target length
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0
TRAIN_STEPS = 2000
LOG_EVERY = 50
SAMPLE_EVERY = 250

# Paths
CORPUS_PATH = Path("data/corpus.txt")

# Checkpoints (`prepare_training_startup` wipes Neo4j first, then may restore)
CHECKPOINT_PATH = Path(
    os.environ.get("CRANE_CHECKPOINT_PATH", "checkpoints/latest.pt")
).resolve()
SAVE_EVERY = max(1, int(os.environ.get("CRANE_SAVE_EVERY", "500")))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "craneneo4j")


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_corpus() -> str:
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"corpus not found at {CORPUS_PATH}. "
            f"Run `uv run python pipelines/fetch_corpus.py` first."
        )
    return CORPUS_PATH.read_text(encoding="utf-8")


def load_tokenizer() -> BPETokenizer:
    tok = BPETokenizer.from_pretrained(TOKENIZER_NAME)
    print(
        f"loaded pretrained tokenizer {TOKENIZER_NAME!r} "
        f"(vocab={tok.vocab_size()}, bos={tok.bos_id}, eos={tok.eos_id})"
    )
    return tok


def build_engine(
    tokenizer: BPETokenizer,
    device: torch.device,
) -> IterativeCognitiveEngine:
    embedder = LookupEmbedder(vocab_size=tokenizer.vocab_size(), embedding_dim=D)
    salient = AttentionPoolSalientExtractor(
        embedding_dim=D, num_slots=SALIENT_SLOTS, num_heads=HEADS
    )
    fusion = CrossAttentionMemoryFusion(embedding_dim=D, num_heads=HEADS)
    reasoner = TransformerReasoner(
        embedding_dim=D, num_layers=REASONER_LAYERS, num_heads=HEADS
    )
    thought_head = LinearThoughtHead(state_dim=D, thought_dim=D)
    decoder = TransformerThoughtDecoder(
        thought_dim=D,
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=D,
        max_len=DECODER_MAX_LEN,
        num_layers=DECODER_LAYERS,
        num_heads=HEADS,
    )

    memory = Neo4jMemoryGraphStore(
        uri=NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), dim=D
    )
    significance = ActivationSignificance()

    config = EngineConfig(
        confidence_threshold=0.85,
        max_iterations=1,             # train short; think() can still iterate
        memory_top_k=16,
        memory_keep_m=8,
        memory_hops=1,
        reinforce_delta=0.05,
        decay_factor=0.9995,
        output_max_len=DECODER_MAX_LEN,
        activation_bump=1.0,
        write_confidence_min=0.0,
        prune_every=500,
        min_edge_weight=1e-3,
        min_node_activation=1e-3,
        feedback_strength=0.1,
    )

    return IterativeCognitiveEngine(
        tokenizer=tokenizer,
        embedder=embedder,
        salient=salient,
        memory=memory,
        significance=significance,
        fusion=fusion,
        reasoner=reasoner,
        thought_head=thought_head,
        decoder=decoder,
        config=config,
        device=device,
    )


def sample_window(
    stream: list[int],
    window: int,
    bos_id: int,
) -> tuple[list[int], list[int]]:
    """Sample a random window from a pre-tokenized stream."""
    max_start = len(stream) - 2 * window - 1
    if max_start <= 0:
        raise ValueError(
            f"corpus too short ({len(stream)} tokens) for window={window}"
        )
    start = random.randint(0, max_start)
    prefix_ids = stream[start : start + window]
    target_ids = [bos_id, *stream[start + window : start + 2 * window]]
    return prefix_ids, target_ids


def score_from_loss(loss_value: float, baseline: float) -> float:
    """Map loss to a feedback score in [-1, +1] via tanh(baseline - loss)."""
    return float(math.tanh(baseline - loss_value))


@torch.no_grad()
def sample_generation(engine: IterativeCognitiveEngine, prompt: str) -> str:
    return engine.think(prompt)


def train() -> None:
    random.seed(0)
    torch.manual_seed(0)

    device = auto_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    text = load_corpus()
    print(f"corpus: {len(text):,} characters")

    tokenizer = load_tokenizer()
    stream = tokenizer.encode_raw(text)
    print(f"stream: {len(stream):,} tokens")

    engine = build_engine(tokenizer, device)
    optimizer = torch.optim.AdamW(
        list(engine.trainable_parameters()), lr=LEARNING_RATE
    )
    total_params = sum(p.numel() for p in engine.trainable_parameters())
    print(f"trainable params: {total_params:,}")

    resumed = prepare_training_startup(
        engine,
        optimizer,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        embedding_dim=D,
        tokenizer_name=TOKENIZER_NAME,
    )
    if resumed:
        print(
            f"checkpoint: resumed from {CHECKPOINT_PATH} "
            f"(global_step={engine.training_step})"
        )
    else:
        print(
            f"checkpoint: fresh run (neo4j wiped; "
            f"no valid file at {CHECKPOINT_PATH})"
        )

    baseline_loss = math.log(tokenizer.vocab_size())
    running_loss = 0.0
    running_n = 0

    started = time.time()
    for _ in range(TRAIN_STEPS):
        prefix_ids, target_ids = sample_window(
            stream, WINDOW_TOKENS, tokenizer.bos_id
        )

        out = engine.train_step_ids(prefix_ids, target_ids, update_memory=True)
        loss: torch.Tensor = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(engine.trainable_parameters()), GRAD_CLIP
        )
        optimizer.step()

        engine.feedback(score_from_loss(loss.item(), baseline_loss))

        running_loss += loss.item() * out["n_tokens"]
        running_n += out["n_tokens"]
        gs = engine.training_step

        if gs % LOG_EVERY == 0:
            avg = running_loss / max(1, running_n)
            ppl = math.exp(min(avg, 20.0))
            elapsed = time.time() - started
            print(
                f"step {gs:>5d} | loss {avg:.4f} | "
                f"ppl {ppl:10.2f} | conf {out['confidence']:.3f} | "
                f"{gs / elapsed:.1f} steps/s"
            )
            running_loss = 0.0
            running_n = 0

        if gs % SAMPLE_EVERY == 0:
            cursor = random.randint(0, len(text) - 100)
            prompt = text[cursor : cursor + 32]
            generated = sample_generation(engine, prompt)
            print(f"  prompt: {prompt!r}")
            print(f"  output: {generated!r}")

        if gs % SAVE_EVERY == 0:
            save_checkpoint(
                CHECKPOINT_PATH,
                engine,
                optimizer,
                embedding_dim=D,
                tokenizer_name=TOKENIZER_NAME,
            )
            print(f"  checkpoint saved -> {CHECKPOINT_PATH}")

    save_checkpoint(
        CHECKPOINT_PATH,
        engine,
        optimizer,
        embedding_dim=D,
        tokenizer_name=TOKENIZER_NAME,
    )
    print(f"training done; final checkpoint -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()

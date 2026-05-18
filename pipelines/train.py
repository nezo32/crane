"""
Next-token prediction training loop for CRANE.

Fast path (most steps): GPU-only — no Neo4j round-trips.
Memory path (every ``CRANE_MEMORY_EVERY`` steps): retrieve, write, feedback,
optional decay/prune.

Usage:
    docker compose up -d
    uv run python pipelines/fetch_corpus.py
    uv run python pipelines/train.py
    uv run tensorboard --logdir runs/crane
"""

import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

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
from corpus_stream import default_cache_path, open_token_stream, sample_window


load_dotenv_if_present()


D = 256
HEADS = 8
SALIENT_SLOTS = 16
REASONER_LAYERS = 4
DECODER_LAYERS = 4
DECODER_MAX_LEN = 128

TOKENIZER_NAME = "gpt2"

WINDOW_TOKENS = 64
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0
TRAIN_STEPS = max(1, int(os.environ.get("CRANE_TRAIN_STEPS", "2000")))
LOG_EVERY = 50
SAMPLE_EVERY = 250
SAMPLE_DECODE_LEN = 48

CORPUS_PATH = Path(os.environ.get("CRANE_CORPUS_PATH", "data/corpus.txt"))
TOKEN_CACHE_PATH = default_cache_path()
REBUILD_TOKEN_CACHE = os.environ.get("CRANE_REBUILD_TOKEN_CACHE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

CHECKPOINT_PATH = Path(
    os.environ.get("CRANE_CHECKPOINT_PATH", "checkpoints/latest.pt")
).resolve()
SAVE_EVERY = max(1, int(os.environ.get("CRANE_SAVE_EVERY", "500")))
MEMORY_EVERY = max(1, int(os.environ.get("CRANE_MEMORY_EVERY", "50")))
DECAY_ON_MEMORY = os.environ.get("CRANE_DECAY_ON_MEMORY", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "craneneo4j")

TENSORBOARD_ENABLED = os.environ.get("CRANE_TENSORBOARD", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
TENSORBOARD_ROOT = Path(
    os.environ.get("CRANE_TENSORBOARD_DIR", "runs/crane")
).resolve()
TENSORBOARD_RUN_NAME = os.environ.get("CRANE_TENSORBOARD_RUN_NAME", "").strip()
TENSORBOARD_LOG_EVERY = max(
    1, int(os.environ.get("CRANE_TENSORBOARD_LOG_EVERY", str(LOG_EVERY)))
)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        max_iterations=1,
        memory_top_k=16,
        memory_keep_m=8,
        memory_hops=1,
        reinforce_delta=0.05,
        decay_factor=0.9995,
        output_max_len=DECODER_MAX_LEN,
        activation_bump=1.0,
        write_confidence_min=0.35,
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


def score_from_loss(loss_value: float, baseline: float) -> float:
    return float(math.tanh(baseline - loss_value))


def tensorboard_log_dir() -> Path:
    run_name = TENSORBOARD_RUN_NAME or datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    return TENSORBOARD_ROOT / run_name


def create_summary_writer(start_step: int) -> SummaryWriter | None:
    if not TENSORBOARD_ENABLED:
        return None
    log_dir = tensorboard_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text(
        "config",
        "\n".join(
            (
                f"embedding_dim={D}",
                f"learning_rate={LEARNING_RATE}",
                f"window_tokens={WINDOW_TOKENS}",
                f"train_steps={TRAIN_STEPS}",
                f"memory_every={MEMORY_EVERY}",
                f"save_every={SAVE_EVERY}",
                f"tokenizer={TOKENIZER_NAME}",
                f"corpus={CORPUS_PATH}",
                f"resumed_global_step={start_step}",
            )
        ),
        global_step=start_step,
    )
    print(f"tensorboard: {log_dir}")
    print(f"  uv run tensorboard --logdir {TENSORBOARD_ROOT}")
    return writer


def train() -> None:
    random.seed(0)
    torch.manual_seed(0)

    device = auto_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"corpus not found at {CORPUS_PATH}. "
            f"Run `uv run python pipelines/fetch_corpus.py` first."
        )
    print(f"corpus file: {CORPUS_PATH} ({CORPUS_PATH.stat().st_size:,} bytes)")

    tokenizer = load_tokenizer()
    stream = open_token_stream(
        CORPUS_PATH,
        tokenizer,
        TOKEN_CACHE_PATH,
        tokenizer_name=TOKENIZER_NAME,
        rebuild=REBUILD_TOKEN_CACHE,
    )
    print(f"train stream: {len(stream):,} tokens (memmap, low RAM)")

    engine = build_engine(tokenizer, device)
    optimizer = torch.optim.AdamW(
        list(engine.trainable_parameters()), lr=LEARNING_RATE
    )
    total_params = sum(p.numel() for p in engine.trainable_parameters())
    print(f"trainable params: {total_params:,}")
    print(
        f"train mode: neo4j every {MEMORY_EVERY} steps; "
        f"decay_on_memory={DECAY_ON_MEMORY}; save every {SAVE_EVERY}"
    )

    resumed = prepare_training_startup(
        engine,
        optimizer,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        embedding_dim=D,
        tokenizer_name=TOKENIZER_NAME,
    )
    start_step = engine.training_step
    if resumed:
        print(
            f"checkpoint: resumed from {CHECKPOINT_PATH} "
            f"(global_step={start_step})"
        )
    else:
        print(
            f"checkpoint: fresh run (neo4j wiped; "
            f"no file at {CHECKPOINT_PATH})"
        )

    baseline_loss = math.log(tokenizer.vocab_size())
    running_loss = 0.0
    running_n = 0

    writer = create_summary_writer(start_step)
    started = time.time()
    try:
        _training_loop(
            engine=engine,
            optimizer=optimizer,
            tokenizer=tokenizer,
            stream=stream,
            start_step=start_step,
            baseline_loss=baseline_loss,
            writer=writer,
            started=started,
        )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    save_checkpoint(
        CHECKPOINT_PATH,
        engine,
        optimizer,
        embedding_dim=D,
        tokenizer_name=TOKENIZER_NAME,
    )
    print(f"training done; final checkpoint -> {CHECKPOINT_PATH}")


def _training_loop(
    *,
    engine: IterativeCognitiveEngine,
    optimizer: torch.optim.Optimizer,
    tokenizer: BPETokenizer,
    stream,
    start_step: int,
    baseline_loss: float,
    writer: SummaryWriter | None,
    started: float,
) -> None:
    running_loss = 0.0
    running_n = 0

    for batch_idx in range(TRAIN_STEPS):
        prefix_ids, target_ids = sample_window(
            stream, WINDOW_TOKENS, tokenizer.bos_id
        )

        gs = start_step + batch_idx + 1
        memory_step = gs % MEMORY_EVERY == 0
        use_memory = memory_step
        update_memory = memory_step
        run_decay = memory_step and DECAY_ON_MEMORY
        run_prune = memory_step and engine._config.prune_every > 0 and (
            gs % engine._config.prune_every == 0
        )

        out = engine.train_step_ids(
            prefix_ids,
            target_ids,
            use_memory=use_memory,
            update_memory=update_memory,
            run_decay=run_decay,
            run_prune=run_prune,
        )
        loss: torch.Tensor = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(engine.trainable_parameters()), GRAD_CLIP
        )
        optimizer.step()

        engine.set_training_step(gs)

        feedback_score = 0.0
        if update_memory:
            feedback_score = score_from_loss(loss.item(), baseline_loss)
            engine.feedback(feedback_score)

        running_loss += loss.item() * out["n_tokens"]
        running_n += out["n_tokens"]

        if writer is not None and gs % TENSORBOARD_LOG_EVERY == 0:
            token_loss = loss.item()
            writer.add_scalar("train/loss_step", token_loss, gs)
            writer.add_scalar("train/confidence", out["confidence"], gs)
            writer.add_scalar("train/grad_norm", float(grad_norm), gs)
            writer.add_scalar("train/memory_step", float(memory_step), gs)
            if update_memory:
                writer.add_scalar("train/feedback_score", feedback_score, gs)

        if gs % LOG_EVERY == 0:
            avg = running_loss / max(1, running_n)
            ppl = math.exp(min(avg, 20.0))
            elapsed = time.time() - started
            mode = "mem" if memory_step else "gpu"
            steps_per_sec = gs / elapsed
            print(
                f"step {gs:>5d} | loss {avg:.4f} | "
                f"ppl {ppl:10.2f} | conf {out['confidence']:.3f} | "
                f"{mode} | {steps_per_sec:.1f} steps/s"
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg, gs)
                writer.add_scalar("train/perplexity", ppl, gs)
                writer.add_scalar("train/steps_per_sec", steps_per_sec, gs)
            running_loss = 0.0
            running_n = 0

        if gs % SAMPLE_EVERY == 0:
            prefix_ids, _ = sample_window(
                stream, WINDOW_TOKENS, tokenizer.bos_id
            )
            prompt = tokenizer.decode(prefix_ids[-24:])
            decoded = engine.decode_from_prefix_ids(
                prefix_ids=prefix_ids,
                max_tokens=SAMPLE_DECODE_LEN,
                use_memory=False,
            )
            print(f"  prompt: {prompt!r}")
            print(f"  decode (train stack): {decoded!r}")
            if writer is not None:
                writer.add_text(
                    "samples/decode",
                    f"**prompt:** {prompt!r}\n\n**decode:** {decoded!r}",
                    gs,
                )

        if gs % SAVE_EVERY == 0:
            save_checkpoint(
                CHECKPOINT_PATH,
                engine,
                optimizer,
                embedding_dim=D,
                tokenizer_name=TOKENIZER_NAME,
            )
            print(f"  checkpoint saved -> {CHECKPOINT_PATH}")
            if writer is not None:
                writer.add_scalar("checkpoint/saved", 1.0, gs)


if __name__ == "__main__":
    train()

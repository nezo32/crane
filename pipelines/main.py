"""
Inference demo for CRANE.

Loads ``checkpoints/latest.pt`` when present (weights + optional Neo4j snapshot).
Uses the multi-iteration ``think()`` loop for generation.
"""

import os
from pathlib import Path

import torch

from crane.config import load_dotenv_if_present
from crane.cognition.engine.base.engine import EngineConfig
from crane.cognition.engine.modules.iterative_engine import IterativeCognitiveEngine
from crane.language.embeddings.modules.lookup_embedder import LookupEmbedder
from crane.language.tokenization.base.tokenizer import Tokenizer
from crane.language.tokenization.modules.bpe_tokenizer import BPETokenizer
from crane.language.tokenization.modules.char_tokenizer import CharTokenizer
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

from checkpoint import load_checkpoint_for_inference


load_dotenv_if_present()


D = 256
HEADS = 8
SALIENT_SLOTS = 16
REASONER_LAYERS = 4
DECODER_LAYERS = 4
DECODER_MAX_LEN = 128

TOKENIZER_NAME = "gpt2"

CHECKPOINT_PATH = Path(
    os.environ.get("CRANE_CHECKPOINT_PATH", "checkpoints/latest.pt")
).resolve()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "craneneo4j")


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer() -> Tokenizer:
    try:
        return BPETokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:
        print(f"failed to load {TOKENIZER_NAME!r} ({exc}); falling back to CharTokenizer")
        return CharTokenizer()


def build_engine(device: torch.device) -> IterativeCognitiveEngine:
    tokenizer = load_tokenizer()

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
        max_iterations=4,
        memory_top_k=16,
        memory_keep_m=8,
        memory_hops=1,
        reinforce_delta=0.05,
        decay_factor=0.999,
        output_max_len=DECODER_MAX_LEN,
        activation_bump=1.0,
        write_confidence_min=0.35,
        prune_every=4,
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


def main() -> None:
    device = auto_device()
    print(f"device: {device}")

    engine = build_engine(device)
    print(
        f"tokenizer: {type(engine._tokenizer).__name__}, "
        f"vocab={engine._tokenizer.vocab_size()}"
    )

    if CHECKPOINT_PATH.is_file():
        step = load_checkpoint_for_inference(
            engine,
            CHECKPOINT_PATH,
            device,
            embedding_dim=D,
            tokenizer_name=TOKENIZER_NAME,
            import_neo4j=True,
        )
        print(f"loaded checkpoint from {CHECKPOINT_PATH} (step={step})")
    else:
        print(f"no checkpoint at {CHECKPOINT_PATH}; using random weights")

    prompt = os.environ.get("CRANE_PROMPT", "To be or not to be")
    output = engine.think(prompt)
    print(f"prompt: {prompt!r}")
    print(f"output: {output!r}")


if __name__ == "__main__":
    main()

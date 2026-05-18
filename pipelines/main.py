import os

from crane.cognition.engine.base.engine import EngineConfig
from crane.cognition.engine.modules.iterative_engine import IterativeCognitiveEngine
from crane.language.embeddings.modules.lookup_embedder import LookupEmbedder
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


D = 64
HEADS = 4
SALIENT_SLOTS = 8
REASONER_LAYERS = 2
DECODER_LAYERS = 2
MAX_OUTPUT_LEN = 64


def build_engine() -> IterativeCognitiveEngine:
    tokenizer = CharTokenizer()
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
        max_len=MAX_OUTPUT_LEN,
        num_layers=DECODER_LAYERS,
        num_heads=HEADS,
    )

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "craneneo4j")

    memory = Neo4jMemoryGraphStore(uri=uri, auth=(user, password), dim=D)
    significance = ActivationSignificance()

    config = EngineConfig(
        confidence_threshold=0.85,
        max_iterations=4,
        memory_top_k=16,
        memory_keep_m=8,
        memory_hops=1,
        reinforce_delta=0.05,
        decay_factor=0.999,
        output_max_len=MAX_OUTPUT_LEN,
        activation_bump=1.0,
        write_confidence_min=0.0,
        prune_every=2,
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
    )


def main() -> None:
    engine = build_engine()
    output = engine.think("hello")
    print(f"output: {output!r}")


if __name__ == "__main__":
    main()

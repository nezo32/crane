from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for the iterative cognitive loop."""

    confidence_threshold: float = 0.85
    max_iterations: int = 8
    memory_top_k: int = 32
    memory_keep_m: int = 8
    memory_hops: int = 2
    reinforce_delta: float = 0.05
    decay_factor: float = 0.999
    output_max_len: int = 128

    # Memory hygiene
    activation_bump: float = 1.0
    """How much to add to a memory unit's activation on each retrieval."""

    write_confidence_min: float = 0.0
    """Iterations with confidence below this do NOT persist their thought."""

    prune_every: int = 0
    """Run `memory.prune` every N global iterations. 0 disables pruning."""

    min_edge_weight: float = 1e-3
    """Edges with weight strictly less than this are deleted on prune."""

    min_node_activation: float = 1e-3
    """Nodes with activation < this AND no strong edges are deleted on prune."""

    max_nodes: int | None = None
    """If set, prune keeps only the top-N nodes by activation."""

    # Prediction-error feedback
    feedback_strength: float = 0.1
    """Magnitude multiplier applied to `score` in `feedback(score)`."""


class CognitiveEngine(ABC):
    """High-level orchestrator that turns input text into output text."""

    @abstractmethod
    def think(self, text: str) -> str:
        raise NotImplementedError

    def feedback(self, score: float) -> None:
        """
        Prediction-error driven update for the LAST `think()` call.

        Positive `score` reinforces associations between the memory units that
        were co-activated; negative `score` weakens them. Default: no-op.

        Override in subclasses to participate in training. Implementations
        should clamp / scale `score` and apply it via the underlying store's
        `reinforce` (with positive or negative delta) and optionally
        `bump_activation`.
        """
        return None


__all__ = [
    "CognitiveEngine",
    "EngineConfig",
]

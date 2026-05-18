from itertools import combinations
from typing import Any

import torch

from crane.language.embeddings.base.embedder import Embedder
from crane.language.tokenization.base.tokenizer import Tokenizer
from crane.memory.graph.base.significance import MemorySignificanceFilter
from crane.memory.graph.base.store import MemoryGraphStore
from crane.neural.decoder.base.decoder import ThoughtLanguageDecoder
from crane.neural.fusion.base.fusion import MemoryFusion
from crane.neural.reasoning.base.reasoner import Reasoner
from crane.neural.salient.base.extractor import SalientExtractor
from crane.neural.thought.base.thought_head import ThoughtHead

from ..base.engine import CognitiveEngine, EngineConfig


class IterativeCognitiveEngine(CognitiveEngine):
    """
    Runs the cognitive loop:

        tokenize -> embed -> salient -> graph retrieve ->
        bump activation -> significance -> fuse -> reason -> thought ->
        (gated write + reinforce + decay + periodic prune) ->
        repeat until confidence >= threshold -> decode

    Each iteration injects the previous thought back as a virtual token so the
    next salient extraction sees the in-progress hypothesis.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedder: Embedder,
        salient: SalientExtractor,
        memory: MemoryGraphStore,
        significance: MemorySignificanceFilter,
        fusion: MemoryFusion,
        reasoner: Reasoner,
        thought_head: ThoughtHead,
        decoder: ThoughtLanguageDecoder,
        config: EngineConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if embedder.embedding_dim != memory.dim:
            raise ValueError(
                f"embedder dim ({embedder.embedding_dim}) must match memory dim ({memory.dim})"
            )

        self._tokenizer = tokenizer
        self._embedder = embedder
        self._salient = salient
        self._memory = memory
        self._significance = significance
        self._fusion = fusion
        self._reasoner = reasoner
        self._thought_head = thought_head
        self._decoder = decoder
        self._config = config or EngineConfig()
        self._device = torch.device(device)

        self._bos_id = getattr(tokenizer, "bos_id", 1)
        self._eos_id = getattr(tokenizer, "eos_id", 2)

        self._global_step = 0
        self._last_used_ids: list[str] = []

        for module in (salient, fusion, reasoner, thought_head, decoder):
            if hasattr(module, "to"):
                module.to(self._device)
            if hasattr(module, "eval"):
                module.eval()
        if hasattr(embedder, "to"):
            embedder.to(self._device)
        if hasattr(embedder, "eval"):
            embedder.eval()

    @torch.no_grad()
    def think(self, text: str) -> str:
        ids = self._tokenizer.encode(text)
        token_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        x = self._embedder.embed_tokens(token_ids)

        thought: torch.Tensor | None = None
        last_used: list[str] = []

        for step in range(self._config.max_iterations):
            x_iter = x if thought is None else self._inject_thought(x, thought)
            salient = self._salient(x_iter)
            query = salient.mean(dim=1).squeeze(0)

            result = self._memory.retrieve(
                query=query.detach().cpu(),
                k=self._config.memory_top_k,
                hops=self._config.memory_hops,
            )
            scores = self._significance.score(query.detach().cpu(), result)

            mem_vectors, mem_weights, used_ids = self._pick_top_m(result, scores)

            if used_ids and self._config.activation_bump > 0:
                self._memory.bump_activation(used_ids, self._config.activation_bump)

            mem_vectors = mem_vectors.to(self._device)
            mem_weights_t = mem_weights.to(self._device)

            fused = self._fusion(
                x_iter,
                mem_vectors.unsqueeze(0),
                mem_weights_t.unsqueeze(0) if mem_weights_t.numel() > 0 else None,
            )
            state = self._reasoner(fused)
            thought, confidence = self._thought_head(state)

            confidence_value = float(confidence.mean().item())
            if confidence_value >= self._config.write_confidence_min:
                self._write_back(thought, used_ids, mem_weights, step)
                last_used = used_ids

            if self._config.decay_factor < 1.0:
                self._memory.decay(self._config.decay_factor)

            self._global_step += 1
            self._maybe_prune()

            if confidence_value >= self._config.confidence_threshold:
                break

        self._last_used_ids = last_used

        if thought is None:
            return ""

        out_ids = self._decoder.generate(
            thought=thought,
            bos_id=self._bos_id,
            eos_id=self._eos_id,
            max_len=self._config.output_max_len,
        )
        return self._tokenizer.decode(out_ids[0].tolist())

    def feedback(self, score: float) -> None:
        """
        Apply prediction-error driven reinforcement to the last `think()` call.

        Positive `score` strengthens associations between memory units that
        co-activated to produce the final thought; negative `score` weakens
        them. The magnitude is `score * config.feedback_strength`.

        No-op if no usable memory was retrieved in the last call.
        """
        if not self._last_used_ids or len(self._last_used_ids) < 2:
            return

        delta = float(score) * self._config.feedback_strength
        if delta == 0.0:
            return

        updates: list[tuple[str, str, float]] = []
        for a, b in combinations(self._last_used_ids, 2):
            updates.append((a, b, delta))
            updates.append((b, a, delta))
        self._memory.reinforce(updates)

        bump = self._config.activation_bump * float(score)
        if bump != 0.0:
            self._memory.bump_activation(self._last_used_ids, bump)

    def _inject_thought(
        self,
        x: torch.Tensor,
        thought: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([thought.unsqueeze(1), x], dim=1)

    def _pick_top_m(
        self,
        result,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        m = self._config.memory_keep_m
        if scores.numel() == 0:
            empty = torch.zeros((0, self._memory.dim), dtype=torch.float32)
            return empty, torch.zeros((0,), dtype=torch.float32), []

        k = min(m, scores.numel())
        top = torch.topk(scores, k=k)
        idx = top.indices.tolist()
        vectors = result.vectors[top.indices]
        weights = scores[top.indices].clamp(min=0.0)
        ids = [result.node_ids[i] for i in idx]
        return vectors.to(torch.float32), weights.to(torch.float32), ids

    def _write_back(
        self,
        thought: torch.Tensor,
        used_ids: list[str],
        used_weights: torch.Tensor,
        step: int,
    ) -> None:
        new_marker = "<new:0>"
        links: list[tuple[str, str, float]] = []

        if used_ids:
            for nid, w in zip(used_ids, used_weights.tolist()):
                links.append((nid, new_marker, float(w)))
                links.append((new_marker, nid, float(w)))

        payload: list[dict[str, Any]] = [{"step": int(step)}]
        thought_cpu = thought.detach().cpu().to(torch.float32)

        self._memory.write(
            vectors=thought_cpu,
            payload=payload,
            links=links,
        )

        if len(used_ids) >= 2 and self._config.reinforce_delta > 0:
            updates: list[tuple[str, str, float]] = []
            for a, b in combinations(used_ids, 2):
                updates.append((a, b, self._config.reinforce_delta))
                updates.append((b, a, self._config.reinforce_delta))
            self._memory.reinforce(updates)

    def _maybe_prune(self) -> None:
        every = self._config.prune_every
        if every <= 0:
            return
        if self._global_step % every != 0:
            return
        self._memory.prune(
            min_edge_weight=self._config.min_edge_weight,
            min_node_activation=self._config.min_node_activation,
            max_nodes=self._config.max_nodes,
        )


__all__ = [
    "IterativeCognitiveEngine",
]

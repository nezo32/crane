from itertools import combinations
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F

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

    Training uses a single cognitive pass with gradients enabled; the iterative
    loop is reserved for inference / test-time compute.
    """

    _TRAINABLE_MODULE_ATTRS = (
        "_embedder",
        "_salient",
        "_fusion",
        "_reasoner",
        "_thought_head",
        "_decoder",
    )

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
        self._pad_id = getattr(tokenizer, "pad_id", 0)

        self._global_step = 0
        self._last_used_ids: list[str] = []

        for module in self._iter_modules():
            module.to(self._device)
        self._set_training(False)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over every learnable parameter the engine owns."""
        for module in self._iter_modules():
            yield from module.parameters()

    @property
    def training_step(self) -> int:
        """Monotonic counter updated during ``train_step*`` memory hygiene."""
        return self._global_step

    def set_training_step(self, n: int) -> None:
        """Use when restoring from a checkpoint (completed optimizer steps)."""
        self._global_step = int(n)

    def module_state_dict(self) -> dict[str, Any]:
        """State dict for all ``nn.Module`` components (CPU tensors)."""
        out: dict[str, Any] = {}
        pairs = (
            ("embedder", "_embedder"),
            ("salient", "_salient"),
            ("fusion", "_fusion"),
            ("reasoner", "_reasoner"),
            ("thought_head", "_thought_head"),
            ("decoder", "_decoder"),
        )
        for key, attr in pairs:
            mod = getattr(self, attr)
            if isinstance(mod, nn.Module):
                out[key] = {k: v.cpu() for k, v in mod.state_dict().items()}
        return out

    def load_module_state_dict(
        self,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        mapping = (
            ("embedder", "_embedder"),
            ("salient", "_salient"),
            ("fusion", "_fusion"),
            ("reasoner", "_reasoner"),
            ("thought_head", "_thought_head"),
            ("decoder", "_decoder"),
        )
        for key, attr in mapping:
            if key not in state:
                continue
            mod = getattr(self, attr)
            if isinstance(mod, nn.Module):
                mod.load_state_dict(state[key], strict=strict)

    @torch.no_grad()
    def think(self, text: str) -> str:
        self._set_training(False)

        ids = self._tokenizer.encode(text)
        token_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        x = self._embedder.embed_tokens(token_ids)

        thought: torch.Tensor | None = None
        last_used: list[str] = []

        for step in range(self._config.max_iterations):
            x_iter = x if thought is None else self._inject_thought(x, thought)
            thought, confidence, used_ids, used_weights = self._cognitive_step(x_iter)

            confidence_value = float(confidence.mean().item())
            if confidence_value >= self._config.write_confidence_min:
                self._write_back(thought.detach(), used_ids, used_weights)
                last_used = used_ids

            if self._config.decay_factor < 1.0:
                self._memory.decay(self._config.decay_factor)

            self._global_step += 1
            every = self._config.prune_every
            if every > 0 and self._global_step % every == 0:
                self.run_memory_prune()

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

    def train_step(
        self,
        input_text: str,
        target_text: str,
        *,
        update_memory: bool = True,
    ) -> dict[str, Any]:
        """
        Text-input variant of `train_step_ids` — encodes both arguments via
        the engine's tokenizer (which adds BOS/EOS), then delegates.
        """
        input_ids = self._tokenizer.encode(input_text)
        target_ids = self._tokenizer.encode(target_text)
        return self.train_step_ids(
            input_ids, target_ids, update_memory=update_memory
        )

    def train_step_ids(
        self,
        input_ids: Sequence[int] | torch.Tensor,
        target_ids: Sequence[int] | torch.Tensor,
        *,
        use_memory: bool = True,
        update_memory: bool = False,
        run_decay: bool = False,
        run_prune: bool = False,
    ) -> dict[str, Any]:
        """
        One supervised training step for next-token prediction.

        - Runs a SINGLE cognitive pass over `input_ids` with gradients enabled.
        - Teacher-forces the decoder on `target_ids` and returns CE loss.
        - Neo4j side effects are opt-in: set ``use_memory=False`` for a fast
          GPU-only path (empty retrieval). Set ``update_memory=True`` only on
          occasional steps to write/reinforce; ``run_decay`` / ``run_prune``
          should be rare (global graph scans are expensive).

        The training loop should call ``set_training_step`` itself each
        optimizer step — this method does not advance ``training_step``.
        """
        self._set_training(True)

        input_tensor = self._to_id_tensor(input_ids)
        target_tensor = self._to_id_tensor(target_ids)

        if target_tensor.size(1) < 2:
            raise ValueError(
                "target_ids must contain at least 2 tokens (e.g. [BOS, t0])"
            )

        x = self._embedder.embed_tokens(input_tensor)
        thought, confidence, used_ids, used_weights = self._cognitive_step(
            x, use_memory=use_memory
        )

        decoder_input = target_tensor[:, :-1]
        decoder_target = target_tensor[:, 1:]

        logits = self._decoder(thought, decoder_input)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1),
            ignore_index=self._pad_id,
        )

        n_tokens = int((decoder_target != self._pad_id).sum().item())
        confidence_value = float(confidence.mean().item())

        if update_memory:
            if confidence_value >= self._config.write_confidence_min:
                self._write_back(thought.detach(), used_ids, used_weights)
                self._last_used_ids = used_ids
            if run_decay and self._config.decay_factor < 1.0:
                self._memory.decay(self._config.decay_factor)
            if run_prune:
                self.run_memory_prune()

        return {
            "loss": loss,
            "logits": logits,
            "n_tokens": n_tokens,
            "confidence": confidence_value,
            "used_ids": used_ids,
        }

    @torch.no_grad()
    def decode_from_prefix_ids(
        self,
        prefix_ids: Sequence[int],
        *,
        max_tokens: int = 48,
        use_memory: bool = False,
    ) -> str:
        """
        Greedy decode using the same single-pass stack as training (not the
        multi-iteration ``think()`` loop). Useful for monitoring train progress.
        """
        self._set_training(False)
        input_tensor = self._to_id_tensor(prefix_ids)
        x = self._embedder.embed_tokens(input_tensor)
        thought, _, _, _ = self._cognitive_step(x, use_memory=use_memory)
        limit = min(max_tokens, self._config.output_max_len)
        out_ids = self._decoder.generate(
            thought=thought,
            bos_id=self._bos_id,
            eos_id=self._eos_id,
            max_len=limit,
        )
        return self._tokenizer.decode(out_ids[0].tolist())

    def _to_id_tensor(
        self,
        ids: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            t = ids.to(self._device, dtype=torch.long)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t
        return torch.tensor([list(ids)], dtype=torch.long, device=self._device)

    def feedback(self, score: float) -> None:
        """
        Apply prediction-error driven reinforcement to the last write.

        Positive `score` strengthens associations between memory units that
        co-activated to produce the final thought; negative `score` weakens
        them. The magnitude is `score * config.feedback_strength`.
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

    def _cognitive_step(
        self,
        x: torch.Tensor,
        *,
        use_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        """
        One forward pass: salient -> [retrieve] -> fuse -> reason -> thought.
        When ``use_memory`` is False, skips Neo4j entirely (empty memory).
        """
        salient = self._salient(x)

        if not use_memory:
            empty = torch.zeros(
                (0, self._memory.dim), dtype=torch.float32, device=self._device
            )
            fused = self._fusion(x, empty.unsqueeze(0), None)
            state = self._reasoner(fused)
            thought, confidence = self._thought_head(state)
            return thought, confidence, [], torch.zeros((0,), dtype=torch.float32)

        query = salient.mean(dim=1).squeeze(0)
        query_cpu = query.detach().cpu()

        result = self._memory.retrieve(
            query=query_cpu,
            k=self._config.memory_top_k,
            hops=self._config.memory_hops,
        )
        scores = self._significance.score(query_cpu, result)
        mem_vectors, mem_weights, used_ids = self._pick_top_m(result, scores)

        if used_ids and self._config.activation_bump > 0:
            self._memory.bump_activation(used_ids, self._config.activation_bump)

        mem_vectors = mem_vectors.to(self._device)
        mem_weights_dev = mem_weights.to(self._device)

        fused = self._fusion(
            x,
            mem_vectors.unsqueeze(0),
            mem_weights_dev.unsqueeze(0) if mem_weights_dev.numel() > 0 else None,
        )
        state = self._reasoner(fused)
        thought, confidence = self._thought_head(state)
        return thought, confidence, used_ids, mem_weights

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
    ) -> None:
        new_marker = "<new:0>"
        links: list[tuple[str, str, float]] = []

        if used_ids:
            for nid, w in zip(used_ids, used_weights.tolist()):
                links.append((nid, new_marker, float(w)))
                links.append((new_marker, nid, float(w)))

        payload: list[dict[str, Any]] = [{"step": int(self._global_step)}]
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

    def run_memory_prune(self) -> dict[str, int] | None:
        """Hard-prune the graph (expensive). Caller decides when to invoke."""
        return self._memory.prune(
            min_edge_weight=self._config.min_edge_weight,
            min_node_activation=self._config.min_node_activation,
            max_nodes=self._config.max_nodes,
        )

    def _iter_modules(self) -> Iterator[nn.Module]:
        for attr in self._TRAINABLE_MODULE_ATTRS:
            module = getattr(self, attr)
            if isinstance(module, nn.Module):
                yield module

    def _set_training(self, mode: bool) -> None:
        for module in self._iter_modules():
            module.train(mode)


__all__ = [
    "IterativeCognitiveEngine",
]

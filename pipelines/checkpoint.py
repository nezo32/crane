"""Training checkpoint save/load — PyTorch weights, optimizer state, Neo4j snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from crane.cognition.engine.modules.iterative_engine import IterativeCognitiveEngine
from crane.memory.graph.modules.neo4j_store import Neo4jMemoryGraphStore


CHECKPOINT_VERSION = 1


def _nested_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _nested_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nested_to_cpu(v) for v in obj]
    return obj


def build_checkpoint_blob(
    engine: IterativeCognitiveEngine,
    optimizer: torch.optim.Optimizer,
    *,
    embedding_dim: int,
    tokenizer_name: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory = engine._memory
    if not isinstance(memory, Neo4jMemoryGraphStore):
        raise TypeError(
            "save_checkpoint requires Neo4jMemoryGraphStore; "
            f"got {type(memory)!r}"
        )
    neo_snap = memory.export_snapshot()
    blob: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "embedding_dim": int(embedding_dim),
        "tokenizer_name": tokenizer_name,
        "global_step": int(engine.training_step),
        "model": engine.module_state_dict(),
        "optimizer": _nested_to_cpu(optimizer.state_dict()),
        "neo4j_snapshot": neo_snap,
    }
    if extra_meta:
        blob["meta"] = extra_meta
    return blob


def save_checkpoint(
    path: Path | str,
    engine: IterativeCognitiveEngine,
    optimizer: torch.optim.Optimizer,
    *,
    embedding_dim: int,
    tokenizer_name: str,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = build_checkpoint_blob(
        engine,
        optimizer,
        embedding_dim=embedding_dim,
        tokenizer_name=tokenizer_name,
        extra_meta=extra_meta,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def prepare_training_startup(
    engine: IterativeCognitiveEngine,
    optimizer: torch.optim.Optimizer,
    *,
    checkpoint_path: Path | str | None,
    device: torch.device,
    embedding_dim: int,
    tokenizer_name: str,
) -> bool:
    """
    Mandatory training prelude:

        1. Wipe Neo4j memory units (and rebuild constraint + vector index).
        2. If ``checkpoint_path`` exists: load neural weights, optimizer state,
           engine ``global_step``, and import the serialized Neo4j graph into
           the now-empty database.
        3. If no checkpoint file: neural weights remain as initialized; graph
           stays empty; ``training_step`` is 0.

    Returns ``True`` if a checkpoint was loaded successfully.
    """
    memory = engine._memory
    if not isinstance(memory, Neo4jMemoryGraphStore):
        raise TypeError(
            "checkpoint training startup requires Neo4jMemoryGraphStore; "
            f"got {type(memory)!r}"
        )

    memory.wipe()

    if not checkpoint_path:
        engine.set_training_step(0)
        return False

    path = Path(checkpoint_path)
    if not path.is_file():
        engine.set_training_step(0)
        return False

    blob = torch.load(path, map_location=device, weights_only=False)

    if int(blob["checkpoint_version"]) != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint_version {blob.get('checkpoint_version')}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    if int(blob["embedding_dim"]) != int(embedding_dim):
        raise ValueError(
            f"checkpoint embedding_dim {blob['embedding_dim']} "
            f"!= current pipeline D {embedding_dim}"
        )
    tn = blob.get("tokenizer_name")
    if tn and tn != tokenizer_name:
        raise ValueError(
            f"checkpoint tokenizer_name {tn!r} "
            f"!= current train TOKENIZER_NAME {tokenizer_name!r}"
        )

    engine.load_module_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    engine.set_training_step(int(blob["global_step"]))

    snap = blob.get("neo4j_snapshot")
    if isinstance(snap, dict) and (snap.get("nodes") or snap.get("edges")):
        memory.import_snapshot(snap)
    elif isinstance(snap, dict) and not snap.get("nodes") and not snap.get("edges"):
        pass

    return True


__all__ = [
    "CHECKPOINT_VERSION",
    "build_checkpoint_blob",
    "prepare_training_startup",
    "save_checkpoint",
]

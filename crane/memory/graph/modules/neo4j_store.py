import json
import uuid
from typing import Any

import torch

from neo4j import GraphDatabase, Driver

from ..base.store import MemoryGraphStore
from ..base.types import MemoryEdge, MemoryNode, RetrievalResult


class Neo4jMemoryGraphStore(MemoryGraphStore):
    """
    Neo4j 5.x backed implementation of `MemoryGraphStore`.

    Schema:
        (:MemoryUnit { id, vector: list[float], payload: str(json),
                       created_at: int, activation: float })
        -[:ASSOCIATED_WITH { weight: float, last_used: int }]->
        (:MemoryUnit { ... })

    A native vector index `memory_vectors` is created on MemoryUnit.vector
    (cosine similarity, dim = `dim`).
    """

    _PLACEHOLDER_PREFIX = "<new:"

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        dim: int,
        *,
        database: str = "neo4j",
        index_name: str = "memory_vectors",
        node_label: str = "MemoryUnit",
        edge_type: str = "ASSOCIATED_WITH",
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=auth)
        self._database = database
        self._dim = dim
        self._index = index_name
        self._label = node_label
        self._edge = edge_type
        self._ensure_schema()

    @property
    def dim(self) -> int:
        return self._dim

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jMemoryGraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                f"CREATE CONSTRAINT {self._label}_id_unique IF NOT EXISTS "
                f"FOR (n:{self._label}) REQUIRE n.id IS UNIQUE"
            )
            session.run(
                f"CREATE VECTOR INDEX {self._index} IF NOT EXISTS "
                f"FOR (n:{self._label}) ON (n.vector) "
                f"OPTIONS {{ indexConfig: {{ "
                f"`vector.dimensions`: $dim, "
                f"`vector.similarity_function`: 'cosine' }} }}",
                dim=self._dim,
            )

    def retrieve(
        self,
        query: torch.Tensor,
        k: int,
        hops: int = 1,
    ) -> RetrievalResult:
        if query.dim() != 1 or query.numel() != self._dim:
            raise ValueError(
                f"query must be a 1-D tensor of size {self._dim}, got shape {tuple(query.shape)}"
            )
        if not isinstance(hops, int):
            raise TypeError("hops must be an int")
        if hops < 0:
            raise ValueError("hops must be >= 0")

        vec = query.detach().cpu().to(torch.float32).tolist()

        with self._driver.session(database=self._database) as session:
            seed_records = list(
                session.run(
                    "CALL db.index.vector.queryNodes($index, $k, $vec) "
                    "YIELD node, score "
                    "RETURN node, score",
                    index=self._index,
                    k=k,
                    vec=vec,
                )
            )

            if not seed_records:
                return self._empty_result()

            seeds_by_id: dict[str, MemoryNode] = {}
            similarities: dict[str, float] = {}
            ordered_ids: list[str] = []

            for record in seed_records:
                node = record["node"]
                nid = node["id"]
                if nid not in seeds_by_id:
                    seeds_by_id[nid] = self._node_from_record(node)
                    ordered_ids.append(nid)
                similarities[nid] = float(record["score"])

            paths = []
            if hops > 0:
                # `hops` is an int (validated above); variable-length bounds
                # must be literals at parse time, so it is interpolated.
                expand_cypher = (
                    f"UNWIND $seed_ids AS sid "
                    f"MATCH (seed:{self._label} {{id: sid}}) "
                    f"OPTIONAL MATCH path = (seed)-[:{self._edge}*1..{hops}]-(m:{self._label}) "
                    f"RETURN path"
                )
                paths = list(
                    session.run(expand_cypher, seed_ids=ordered_ids)
                )

        for record in paths:
            path = record["path"]
            if path is None:
                continue
            for node in path.nodes:
                nid = node["id"]
                if nid not in seeds_by_id:
                    seeds_by_id[nid] = self._node_from_record(node)
                    ordered_ids.append(nid)
                    similarities.setdefault(nid, 0.0)

        edges: list[MemoryEdge] = []
        seen_edges: set[tuple[str, str, float, int]] = set()
        for record in paths:
            path = record["path"]
            if path is None:
                continue
            for rel in path.relationships:
                edge = self._edge_from_record(rel)
                key = (edge.source_id, edge.target_id, edge.weight, edge.last_used)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(edge)

        nodes_list = [seeds_by_id[nid] for nid in ordered_ids]
        vectors = torch.stack([n.vector for n in nodes_list], dim=0)
        sims = torch.tensor(
            [similarities[nid] for nid in ordered_ids], dtype=torch.float32
        )

        return RetrievalResult(
            nodes=nodes_list,
            edges=edges,
            vectors=vectors,
            similarities=sims,
            node_ids=ordered_ids,
        )

    def write(
        self,
        vectors: torch.Tensor,
        payload: list[dict[str, Any]],
        links: list[tuple[str, str, float]],
    ) -> list[str]:
        if vectors.dim() != 2 or vectors.size(1) != self._dim:
            raise ValueError(
                f"vectors must be (B, {self._dim}), got {tuple(vectors.shape)}"
            )
        if len(payload) != vectors.size(0):
            raise ValueError("payload length must match number of vectors")

        new_ids = [str(uuid.uuid4()) for _ in range(vectors.size(0))]
        rows = [
            {
                "id": new_ids[i],
                "vector": vectors[i].detach().cpu().to(torch.float32).tolist(),
                "payload": json.dumps(payload[i]),
            }
            for i in range(vectors.size(0))
        ]

        resolved_links: list[dict[str, Any]] = []
        for src, dst, weight in links:
            src_id = self._resolve_marker(src, new_ids)
            dst_id = self._resolve_marker(dst, new_ids)
            if src_id is None or dst_id is None:
                continue
            resolved_links.append(
                {"src": src_id, "dst": dst_id, "weight": float(weight)}
            )

        create_nodes = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{self._label} {{ id: row.id }}) "
            f"ON CREATE SET n.vector = row.vector, "
            f"              n.payload = row.payload, "
            f"              n.created_at = timestamp(), "
            f"              n.activation = 0.0"
        )

        create_edges = (
            f"UNWIND $links AS l "
            f"MATCH (a:{self._label} {{ id: l.src }}) "
            f"MATCH (b:{self._label} {{ id: l.dst }}) "
            f"MERGE (a)-[r:{self._edge}]->(b) "
            f"ON CREATE SET r.weight = l.weight, r.last_used = timestamp() "
            f"ON MATCH  SET r.weight = r.weight + l.weight, r.last_used = timestamp()"
        )

        with self._driver.session(database=self._database) as session:
            session.run(create_nodes, rows=rows)
            if resolved_links:
                session.run(create_edges, links=resolved_links)

        return new_ids

    def reinforce(
        self,
        edge_updates: list[tuple[str, str, float]],
    ) -> None:
        if not edge_updates:
            return

        updates = [
            {"src": s, "dst": d, "delta": float(w)} for s, d, w in edge_updates
        ]

        cypher = (
            f"UNWIND $updates AS u "
            f"MATCH (a:{self._label} {{ id: u.src }}) "
            f"MATCH (b:{self._label} {{ id: u.dst }}) "
            f"MERGE (a)-[r:{self._edge}]->(b) "
            f"ON CREATE SET r.weight = u.delta, r.last_used = timestamp() "
            f"ON MATCH  SET r.weight = r.weight + u.delta, r.last_used = timestamp()"
        )

        with self._driver.session(database=self._database) as session:
            session.run(cypher, updates=updates)

    def bump_activation(
        self,
        node_ids: list[str],
        delta: float,
    ) -> None:
        if not node_ids:
            return

        cypher = (
            f"UNWIND $ids AS nid "
            f"MATCH (n:{self._label} {{ id: nid }}) "
            f"SET n.activation = coalesce(n.activation, 0.0) + $delta"
        )

        with self._driver.session(database=self._database) as session:
            session.run(cypher, ids=list(node_ids), delta=float(delta))

    def decay(self, factor: float) -> None:
        if factor < 0:
            raise ValueError("decay factor must be >= 0")

        edge_cypher = (
            f"MATCH ()-[r:{self._edge}]->() "
            f"SET r.weight = r.weight * $factor"
        )
        node_cypher = (
            f"MATCH (n:{self._label}) "
            f"SET n.activation = coalesce(n.activation, 0.0) * $factor"
        )

        with self._driver.session(database=self._database) as session:
            session.run(edge_cypher, factor=float(factor))
            session.run(node_cypher, factor=float(factor))

    def prune(
        self,
        *,
        min_edge_weight: float,
        min_node_activation: float,
        max_nodes: int | None = None,
    ) -> dict[str, int]:
        edges_deleted = 0
        nodes_deleted = 0

        drop_edges = (
            f"MATCH ()-[r:{self._edge}]->() "
            f"WHERE r.weight < $threshold "
            f"WITH r, count(*) AS _ "
            f"DELETE r "
            f"RETURN count(*) AS deleted"
        )

        drop_cold_orphans = (
            f"MATCH (n:{self._label}) "
            f"WHERE coalesce(n.activation, 0.0) < $threshold "
            f"  AND NOT EXISTS {{ "
            f"      MATCH (n)-[r:{self._edge}]-() "
            f"      WHERE r.weight >= $edge_threshold "
            f"  }} "
            f"WITH n "
            f"DETACH DELETE n "
            f"RETURN count(*) AS deleted"
        )

        cap_nodes = (
            f"MATCH (n:{self._label}) "
            f"WITH n ORDER BY coalesce(n.activation, 0.0) DESC SKIP $keep "
            f"DETACH DELETE n "
            f"RETURN count(*) AS deleted"
        )

        with self._driver.session(database=self._database) as session:
            rec = session.run(drop_edges, threshold=float(min_edge_weight)).single()
            if rec is not None:
                edges_deleted = int(rec["deleted"])

            rec = session.run(
                drop_cold_orphans,
                threshold=float(min_node_activation),
                edge_threshold=float(min_edge_weight),
            ).single()
            if rec is not None:
                nodes_deleted = int(rec["deleted"])

            if max_nodes is not None and max_nodes >= 0:
                rec = session.run(cap_nodes, keep=int(max_nodes)).single()
                if rec is not None:
                    nodes_deleted += int(rec["deleted"])

        return {"edges_deleted": edges_deleted, "nodes_deleted": nodes_deleted}

    def _empty_result(self) -> RetrievalResult:
        return RetrievalResult(
            nodes=[],
            edges=[],
            vectors=torch.zeros((0, self._dim), dtype=torch.float32),
            similarities=torch.zeros((0,), dtype=torch.float32),
            node_ids=[],
        )

    def _node_from_record(self, n: Any) -> MemoryNode:
        payload_raw = n.get("payload") if hasattr(n, "get") else n["payload"]
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except (TypeError, json.JSONDecodeError):
            payload = {}
        vector = torch.tensor(n["vector"], dtype=torch.float32)
        activation = float(n["activation"]) if "activation" in n else 0.0
        return MemoryNode(
            id=n["id"],
            vector=vector,
            payload=payload,
            activation=activation,
        )

    def _edge_from_record(self, r: Any) -> MemoryEdge:
        return MemoryEdge(
            source_id=r.nodes[0]["id"],
            target_id=r.nodes[1]["id"],
            weight=float(r.get("weight", 0.0)),
            last_used=int(r.get("last_used", 0) or 0),
        )

    def _resolve_marker(
        self,
        marker: str,
        new_ids: list[str],
    ) -> str | None:
        if not marker.startswith(self._PLACEHOLDER_PREFIX):
            return marker
        try:
            idx = int(marker[len(self._PLACEHOLDER_PREFIX) : -1])
        except ValueError:
            return None
        if 0 <= idx < len(new_ids):
            return new_ids[idx]
        return None


__all__ = [
    "Neo4jMemoryGraphStore",
]

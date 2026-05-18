"""
Wipe the Neo4j memory graph completely.

Uses ``Neo4jMemoryGraphStore.wipe()`` for the CRANE subtree so constraint +
vector index are dropped and recreated (matches training startup).

Default (CRANE-only)::

    uv run python pipelines/wipe_memory.py

Wipe entire database (every label — use with care)::

    uv run python pipelines/wipe_memory.py --all

``CRANE_EMBEDDING_DIM`` (default ``256``) must match [`pipelines/train.py`](train.py)
embedding dim when recreating the vector index dimension.
"""

import os

from neo4j import GraphDatabase

from crane.config import load_dotenv_if_present


load_dotenv_if_present()


NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "craneneo4j")

NODE_LABEL = "MemoryUnit"
INDEX_NAME = "memory_vectors"
CONSTRAINT_NAME = f"{NODE_LABEL}_id_unique"


def main() -> None:
    import sys

    wipe_all = "--all" in sys.argv

    print(f"connecting to {NEO4J_URI} as {NEO4J_USER}")
    if not wipe_all:
        dim = int(os.environ.get("CRANE_EMBEDDING_DIM", "256"))
        from crane.memory.graph.modules.neo4j_store import Neo4jMemoryGraphStore

        store = Neo4jMemoryGraphStore(
            uri=NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            dim=dim,
        )
        store.wipe()
        store.close()
        print("Neo4j CRANE memory wiped; constraint + vector index recreated.")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            total = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            print(f"{total:,} nodes in database; deleting all nodes...")
            session.run("MATCH (n) DETACH DELETE n")
            session.run(f"DROP INDEX {INDEX_NAME} IF EXISTS")
            session.run(f"DROP CONSTRAINT {CONSTRAINT_NAME} IF EXISTS")
            print("wipe complete (full database). Neo4j index/constraint dropped; recreate on next app connect.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()

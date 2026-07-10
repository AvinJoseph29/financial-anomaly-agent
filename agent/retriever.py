"""
Retriever Node
==============
Handles both "retrieve" (vector search) and "graph" (Neo4j traversal) routes.

For "retrieve": embeds the query, searches Qdrant, optionally filtered by company.
For "graph":    runs a Cypher query against Neo4j based on the query intent.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from neo4j import GraphDatabase

from agent.state import AgentState

load_dotenv()

QDRANT_PATH    = Path(os.getenv("QDRANT_PATH", "./data/qdrant_store"))
COLLECTION     = "sec_filings"
EMBED_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

TOP_K = 5


# ── Lazy singletons — loaded once, reused across calls ──────────────────────

@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_qdrant() -> QdrantClient:
    return QdrantClient(path=str(QDRANT_PATH))


@lru_cache(maxsize=1)
def _get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Vector retrieval ──────────────────────────────────────────────────────

def vector_search(query: str, company_key: Optional[str] = None, top_k: int = TOP_K) -> list:
    """Embed the query and search Qdrant, optionally filtered by company."""
    model  = _get_embedder()
    client = _get_qdrant()

    vector = model.encode(query, normalize_embeddings=True).tolist()

    query_filter = None
    if company_key:
        query_filter = Filter(
            must=[FieldCondition(key="company_key", match=MatchValue(value=company_key))]
        )

    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "text":         p.payload["text"],
            "company_name": p.payload["company_name"],
            "ticker":       p.payload["ticker"],
            "period":       p.payload["period"],
            "source_url":   p.payload.get("source_url", ""),
            "score":        round(p.score, 3),
        }
        for p in results
    ]


# ── Graph retrieval ───────────────────────────────────────────────────────

GRAPH_QUERY_TEMPLATES = {
    "auditor": """
        MATCH (c:Company {key: $company_key})-[r:AUDITED_BY]->(a:Auditor)
        RETURN c.name AS company, a.name AS auditor, a.note AS auditor_note, r.years AS years
    """,
    "subsidiaries": """
        MATCH (c:Company {key: $company_key})-[:HAS_SUBSIDIARY]->(s:Subsidiary)
        RETURN c.name AS company, s.name AS subsidiary, s.type AS type, s.note AS note
        ORDER BY s.name
    """,
    "risks": """
        MATCH (c:Company {key: $company_key})-[:HAS_RISK]->(r:RiskFactor)
        RETURN c.name AS company, r.name AS risk, r.severity AS severity, r.note AS note
        ORDER BY r.severity DESC
    """,
    "full_profile": """
        MATCH (c:Company {key: $company_key})
        OPTIONAL MATCH (c)-[:AUDITED_BY]->(a:Auditor)
        OPTIONAL MATCH (c)-[:HAS_SUBSIDIARY]->(s:Subsidiary)
        OPTIONAL MATCH (c)-[:HAS_RISK]->(r:RiskFactor)
        RETURN c.name AS company, c.sector AS sector, c.status AS status,
               collect(DISTINCT a.name) AS auditors,
               collect(DISTINCT s.name) AS subsidiaries,
               collect(DISTINCT r.name) AS risks
    """,
}


def classify_graph_intent(query: str) -> str:
    """Pick which Cypher template fits the query best."""
    q = query.lower()
    if any(w in q for w in ["auditor", "audited", "audit"]):
        return "auditor"
    if any(w in q for w in ["subsidiary", "subsidiaries", "owns", "owned"]):
        return "subsidiaries"
    if any(w in q for w in ["risk", "risks", "danger"]):
        return "risks"
    return "full_profile"


def graph_search(query: str, company_key: Optional[str]) -> list:
    """Run a Cypher query against Neo4j based on query intent."""
    if not company_key:
        return [{"error": "No company specified — graph queries need a target company"}]

    intent   = classify_graph_intent(query)
    cypher   = GRAPH_QUERY_TEMPLATES[intent]
    driver   = _get_neo4j_driver()

    with driver.session() as session:
        result = session.run(cypher, company_key=company_key)
        return result.data()


# ── Node entry point ─────────────────────────────────────────────────────

def retriever_node(state: AgentState) -> dict:
    """
    Dispatches to vector_search or graph_search based on state['route'].
    Both populate the state so the responder can synthesise an answer.
    """
    query       = state["query"]
    company_key = state.get("company_key")
    route       = state.get("route", "retrieve")

    if route == "graph":
        results = graph_search(query, company_key)
        return {"graph_results": results, "vector_results": []}

    # default: vector retrieval
    results = vector_search(query, company_key=company_key)
    return {"vector_results": results, "graph_results": []}
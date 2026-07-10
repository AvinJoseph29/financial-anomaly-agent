"""
Agent State Schema
==================
Shared state that flows through every node in the LangGraph state machine.
"""

from typing import TypedDict, Optional, Literal


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    query: str

    # ── Planner output ────────────────────────────────────────────────────
    route:         Optional[str]   # "retrieve" | "calculate" | "graph" | "direct"
    company_key:   Optional[str]   # extracted target company, e.g. "enron"
    year:          Optional[str]   # extracted target year, e.g. "2000"
    calc_type:     Optional[str]   # "altman_z_score" | "current_ratio" | "debt_to_equity"

    # ── Retriever output ──────────────────────────────────────────────────
    vector_results: list           # chunks from Qdrant
    graph_results:  list           # rows from Neo4j

    # ── Tool output ───────────────────────────────────────────────────────
    calc_result:    Optional[dict]

    # ── Responder output ──────────────────────────────────────────────────
    final_answer:   Optional[str]
    citations:      list

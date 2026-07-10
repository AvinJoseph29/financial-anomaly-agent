"""
Layer 5 — FastAPI Server with Streaming
=========================================
Exposes the LangGraph agent as a REST API with Server-Sent Events streaming.

Endpoints:
    GET  /              — health check
    GET  /companies      — list available companies in the knowledge base
    POST /investigate     — run a query through the agent (blocking, full response)
    POST /investigate/stream — same, but streams progress as SSE events

Usage:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

import os, sys, json, time
from pathlib import Path

# Allow running as `uvicorn api.main:app` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.graph import run_agent, get_app
from agent.state import AgentState


app = FastAPI(
    title="Financial Anomaly Investigation Agent",
    description=(
        "Agentic RAG system over SEC 10-K filings (Apple, SVB, Enron). "
        "Routes queries to vector search, knowledge graph traversal, or "
        "financial ratio calculation depending on intent."
    ),
    version="1.0.0",
)

# CORS — open for local dev / portfolio demo. Tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class InvestigateRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, examples=[
        "What is the Altman Z-score for Enron in 2000?"
    ])


class CitationOut(BaseModel):
    company: str
    period: str
    source_url: str


class InvestigateResponse(BaseModel):
    query: str
    route: str
    answer: str
    citations: list[CitationOut]
    latency_ms: int


class CompanyInfo(BaseModel):
    key: str
    name: str
    ticker: str
    sector: str
    status: str


# ── Routes ────────────────────────────────────────────────────────────────────


def _build_graph_data(state: dict) -> dict:
    """
    Convert Neo4j graph_results rows into D3-compatible nodes + edges.
    Handles auditor, subsidiary, risk, and full_profile query shapes.
    """
    if state.get("route") != "graph":
        return {"nodes": [], "edges": []}

    rows = state.get("graph_results") or []
    nodes, edges = {}, []

    for row in rows:
        # auditor query
        if "auditor" in row and "company" in row:
            co, au = row["company"], row["auditor"]
            nodes.setdefault(co, {"id": co, "label": co, "type": "Company"})
            nodes.setdefault(au, {"id": au, "label": au, "type": "Auditor",
                                   "note": row.get("auditor_note", ""),
                                   "years": row.get("years", "")})
            edges.append({"source": co, "target": au, "label": "AUDITED_BY"})

        # subsidiaries query
        elif "subsidiary" in row and "company" in row:
            co, sub = row["company"], row["subsidiary"]
            nodes.setdefault(co, {"id": co, "label": co, "type": "Company"})
            nodes.setdefault(sub, {"id": sub, "label": sub, "type": "Subsidiary",
                                    "subtype": row.get("type", ""),
                                    "note": row.get("note", "")})
            edges.append({"source": co, "target": sub, "label": "HAS_SUBSIDIARY"})

        # risks query
        elif "risk" in row and "company" in row:
            co, risk = row["company"], row["risk"]
            nodes.setdefault(co, {"id": co, "label": co, "type": "Company"})
            nodes.setdefault(risk, {"id": risk, "label": risk, "type": "RiskFactor",
                                     "severity": row.get("severity", ""),
                                     "note": row.get("note", "")})
            edges.append({"source": co, "target": risk, "label": "HAS_RISK"})

        # full_profile query — flattened lists in single row
        elif "auditors" in row or "subsidiaries" in row:
            co = row.get("company", "Unknown")
            nodes.setdefault(co, {"id": co, "label": co, "type": "Company",
                                   "sector": row.get("sector", ""),
                                   "status": row.get("status", "")})
            for au in (row.get("auditors") or []):
                if au:
                    nodes.setdefault(au, {"id": au, "label": au, "type": "Auditor"})
                    edges.append({"source": co, "target": au, "label": "AUDITED_BY"})
            for sub in (row.get("subsidiaries") or []):
                if sub:
                    nodes.setdefault(sub, {"id": sub, "label": sub, "type": "Subsidiary"})
                    edges.append({"source": co, "target": sub, "label": "HAS_SUBSIDIARY"})
            for risk in (row.get("risks") or []):
                if risk:
                    nodes.setdefault(risk, {"id": risk, "label": risk, "type": "RiskFactor"})
                    edges.append({"source": co, "target": risk, "label": "HAS_RISK"})

    return {"nodes": list(nodes.values()), "edges": edges}


@app.get("/", tags=["health"])
def health_check():
    return {
        "status": "ok",
        "service": "financial-anomaly-investigation-agent",
        "endpoints": ["/investigate", "/investigate/stream", "/companies", "/docs"],
    }


@app.get("/companies", response_model=list[CompanyInfo], tags=["metadata"])
def list_companies():
    """Static list of companies currently in the knowledge base."""
    return [
        {"key": "apple", "name": "Apple Inc.", "ticker": "AAPL",
         "sector": "Technology", "status": "active"},
        {"key": "svb",   "name": "SVB Financial Group", "ticker": "SIVB",
         "sector": "Banking", "status": "failed_2023"},
        {"key": "enron", "name": "Enron Corp", "ticker": "ENE",
         "sector": "Energy", "status": "bankrupt_2001"},
    ]


@app.post("/investigate", response_model=InvestigateResponse, tags=["agent"])
def investigate(req: InvestigateRequest):
    """
    Run a query through the full agent pipeline (blocking).
    Returns once the complete investigation memo is ready.
    """
    start = time.time()
    try:
        result = run_agent(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

    latency_ms = int((time.time() - start) * 1000)

    citations = [
        CitationOut(
            company=c.get("company", "Unknown"),
            period=c.get("period", "N/A"),
            source_url=c.get("source_url", ""),
        )
        for c in result.get("citations", [])
    ]

    return InvestigateResponse(
        query=req.query,
        route=result.get("route", "unknown"),
        answer=result.get("final_answer", "No answer generated."),
        citations=citations,
        latency_ms=latency_ms,
    )


@app.post("/investigate/stream", tags=["agent"])
async def investigate_stream(req: InvestigateRequest):
    """
    Run a query through the agent, streaming progress as Server-Sent Events.

    Event sequence:
      1. {"stage": "planning"}              — planner is deciding the route
      2. {"stage": "routed", "route": "..."} — route decision made
      3. {"stage": "gathering"}             — retriever/calculator running
      4. {"stage": "synthesising"}          — responder is writing the memo
      5. {"stage": "complete", "answer": "...", "citations": [...]}
      6. {"stage": "error", "detail": "..."} — on failure
    """

    async def event_generator():
        try:
            yield {"event": "progress", "data": json.dumps({"stage": "planning"})}

            graph_app = get_app()
            initial_state: AgentState = {
                "query":          req.query,
                "route":          None,
                "company_key":    None,
                "year":           None,
                "calc_type":      None,
                "vector_results": [],
                "graph_results":  [],
                "calc_result":    None,
                "final_answer":   None,
                "citations":      [],
            }

            # LangGraph streams node-by-node. We accumulate ALL node outputs
            # into full_state because graph_results lives in the retriever node
            # while final_answer lives in the responder node — we need both.
            full_state = dict(initial_state)
            for step_output in graph_app.stream(initial_state):
                node_name = list(step_output.keys())[0]
                node_data = step_output[node_name]

                # Merge this node's output into the accumulated state
                full_state.update(node_data)

                if node_name == "planner":
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "stage": "routed",
                            "route": node_data.get("route", "unknown"),
                        }),
                    }
                elif node_name in ("retriever", "calculator"):
                    yield {"event": "progress", "data": json.dumps({"stage": "gathering"})}
                elif node_name == "responder":
                    yield {"event": "progress", "data": json.dumps({"stage": "synthesising"})}

            final_state = full_state

            citations = [
                {
                    "company":    c.get("company", "Unknown"),
                    "period":     c.get("period", "N/A"),
                    "source_url": c.get("source_url", ""),
                }
                for c in final_state.get("citations", [])
            ]

            yield {
                "event": "complete",
                "data": json.dumps({
                    "stage":      "complete",
                    "answer":     final_state.get("final_answer", "No answer generated."),
                    "citations":  citations,
                    "graph_data": _build_graph_data(final_state),
                }),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"stage": "error", "detail": str(e)}),
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
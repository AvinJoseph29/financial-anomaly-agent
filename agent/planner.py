"""
Planner Node
============
First node in the agent. Decides how to handle the incoming query:

  "retrieve"  → needs document search (vector store)
  "graph"     → needs entity relationship traversal (Neo4j)
  "calculate" → needs a financial ratio computed
  "direct"    → can be answered without tools (rare — mostly clarifications)

Uses the LLM with JSON mode to extract: route, company_key, year, calc_type.
This is the node that makes the agent "agentic" rather than a fixed pipeline.
"""

from agent.llm import chat_json
from agent.state import AgentState

KNOWN_COMPANIES = {
    "apple": ["apple", "aapl"],
    "svb":   ["svb", "silicon valley bank", "sivb"],
    "enron": ["enron", "ene"],
}

PLANNER_SYSTEM_PROMPT = """You are the planning module of a financial anomaly investigation agent.

Given a user query about SEC filings (Apple, SVB Financial Group, Enron), decide:

1. route — one of:
   - "calculate"  if the query asks for a financial ratio, score, or computed metric
                  (e.g. "Altman Z-score", "debt to equity", "current ratio", "bankruptcy risk")
   - "graph"      if the query asks about RELATIONSHIPS: auditors, subsidiaries, risk factors,
                  "who audited", "what subsidiaries", "related entities"
   - "retrieve"   if the query asks about NARRATIVE CONTENT in the filings themselves
                  (e.g. "what does the filing say about...", "describe the risk factors mentioned")
   - "direct"     if the query is a greeting or doesn't require any company data

2. company_key — one of "apple", "svb", "enron", or null if not specified/ambiguous
3. year — the 4-digit year mentioned, or null if not specified
4. calc_type — only if route is "calculate": one of "altman_z_score", "current_ratio", "debt_to_equity"

Respond ONLY with JSON in this exact shape:
{"route": "...", "company_key": "...", "year": "...", "calc_type": "..."}

Examples:
Query: "What is the Altman Z-score for Enron in 2000?"
{"route": "calculate", "company_key": "enron", "year": "2000", "calc_type": "altman_z_score"}

Query: "Who audited SVB and what happened to them?"
{"route": "graph", "company_key": "svb", "year": null, "calc_type": null}

Query: "What does Apple's 10-K say about supply chain risk?"
{"route": "retrieve", "company_key": "apple", "year": null, "calc_type": null}

Query: "What subsidiaries did Enron have?"
{"route": "graph", "company_key": "enron", "year": null, "calc_type": null}
"""


def planner_node(state: AgentState) -> dict:
    """
    Determine the routing strategy for this query using the LLM.
    Falls back to keyword matching if the LLM call fails (resilience).
    """
    query = state["query"]

    try:
        result = chat_json([
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ])

        if "error" in result:
            raise ValueError(result.get("raw", "planner LLM error"))

        return {
            "route":       result.get("route", "retrieve"),
            "company_key": result.get("company_key"),
            "year":        result.get("year"),
            "calc_type":   result.get("calc_type"),
        }

    except Exception:
        # Fallback: simple keyword heuristic if the LLM/JSON parsing fails
        return _fallback_route(query)


def _fallback_route(query: str) -> dict:
    """Keyword-based fallback routing — only used if the LLM call fails."""
    q = query.lower()

    company_key = None
    for key, aliases in KNOWN_COMPANIES.items():
        if any(alias in q for alias in aliases):
            company_key = key
            break

    calc_words = ["z-score", "altman", "ratio", "debt to equity", "current ratio", "bankruptcy"]
    graph_words = ["auditor", "audited", "subsidiary", "subsidiaries", "related", "owns"]

    if any(w in q for w in calc_words):
        route = "calculate"
        calc_type = "altman_z_score" if "z-score" in q or "altman" in q else \
                    "debt_to_equity" if "debt" in q else "current_ratio"
    elif any(w in q for w in graph_words):
        route, calc_type = "graph", None
    else:
        route, calc_type = "retrieve", None

    # crude year extraction
    import re
    year_match = re.search(r"\b(19|20)\d{2}\b", q)
    year = year_match.group(0) if year_match else None

    return {"route": route, "company_key": company_key, "year": year, "calc_type": calc_type}

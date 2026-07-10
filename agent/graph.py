"""
Agent Graph — LangGraph State Machine
========================================
Wires together: Planner -> [Retriever | Calculator] -> Responder

This is the core of the agentic pipeline. The Planner decides the route,
then either the Retriever (vector/graph search) or Calculator (financial
ratios) gathers evidence, and the Responder synthesises the final answer.

Usage:
    from agent.graph import run_agent
    result = run_agent("What is the Altman Z-score for Enron in 2000?")
    print(result["final_answer"])
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.planner import planner_node
from agent.retriever import retriever_node
from agent.calculator_node import calculator_node
from agent.responder import responder_node


def route_decision(state: AgentState) -> str:
    """Conditional edge function — reads the planner's route decision."""
    route = state.get("route", "retrieve")
    if route == "calculate":
        return "calculate"
    elif route == "graph":
        return "graph"
    elif route == "direct":
        return "direct"
    return "retrieve"


def build_agent_graph():
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("planner",    planner_node)
    graph.add_node("retriever",  retriever_node)   # handles both "retrieve" and "graph"
    graph.add_node("calculator", calculator_node)
    graph.add_node("responder",  responder_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_decision,
        {
            "retrieve":  "retriever",
            "graph":     "retriever",     # same node, different internal logic
            "calculate": "calculator",
            "direct":    "responder",     # skip straight to responder
        },
    )

    graph.add_edge("retriever",  "responder")
    graph.add_edge("calculator", "responder")
    graph.add_edge("responder",  END)

    return graph.compile()


# Compiled once at import time — reused across requests
_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_agent_graph()
    return _app


def run_agent(query: str) -> dict:
    """
    Run a single query through the full agent pipeline.
    Returns the final state including final_answer and citations.
    """
    app = get_app()
    initial_state: AgentState = {
        "query":           query,
        "route":           None,
        "company_key":     None,
        "year":            None,
        "calc_type":       None,
        "vector_results":  [],
        "graph_results":   [],
        "calc_result":     None,
        "final_answer":    None,
        "citations":       [],
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What is the Altman Z-score for Enron in 2000?"
    print(f"\nQuery: {query}\n")
    result = run_agent(query)
    print("─" * 70)
    print(result["final_answer"])
    print("─" * 70)
    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  • {c['company']} ({c['period']}) — {c['source_url']}")

"""
Responder Node
==============
Final node. Synthesises everything gathered (vector chunks, graph results,
or calculation output) into a coherent investigation memo with citations.
"""

import json
from agent.llm import chat
from agent.state import AgentState


RESPONDER_SYSTEM_PROMPT = """You are a financial investigation analyst writing a brief memo.

Rules:
- Base your answer ONLY on the provided evidence. Never invent figures or facts.
- Cite sources inline using the format [Source: Company Name, Period].
- If the evidence is a calculation, explain what the number means in plain language.
- If the evidence is graph relationships, describe them clearly and factually.
- If the evidence is document excerpts, synthesise them — don't just repeat them verbatim.
- Keep the tone professional and analytical, like an auditor's memo.
- If evidence is insufficient to answer, say so plainly rather than guessing.
- Be concise — 150-300 words unless the query demands more detail.
"""


def _format_vector_evidence(results: list[dict]) -> str:
    if not results:
        return "No relevant document excerpts found."
    blocks = []
    for r in results:
        blocks.append(
            f"[{r['company_name']} {r['period']}] (relevance: {r['score']})\n{r['text'][:800]}"
        )
    return "\n\n---\n\n".join(blocks)


def _format_graph_evidence(results: list[dict]) -> str:
    if not results:
        return "No graph relationships found."
    if "error" in results[0]:
        return results[0]["error"]
    return json.dumps(results, indent=2)


def _format_calc_evidence(calc: dict) -> str:
    if not calc:
        return "No calculation performed."
    if "error" in calc:
        return calc["error"]
    return json.dumps(calc, indent=2)


def responder_node(state: AgentState) -> dict:
    query = state["query"]
    route = state.get("route", "retrieve")

    if route == "calculate":
        evidence = _format_calc_evidence(state.get("calc_result"))
        evidence_label = "CALCULATION RESULT"
    elif route == "graph":
        evidence = _format_graph_evidence(state.get("graph_results", []))
        evidence_label = "GRAPH RELATIONSHIPS"
    else:
        evidence = _format_vector_evidence(state.get("vector_results", []))
        evidence_label = "DOCUMENT EXCERPTS"

    user_message = f"""QUESTION: {query}

{evidence_label}:
{evidence}

Write the investigation memo answering the question above, citing the evidence provided."""

    answer = chat(
        messages=[
            {"role": "system", "content": RESPONDER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
    )

    # Build citation list for transparency
    citations = []
    if route == "retrieve":
        citations = [
            {"company": r["company_name"], "period": r["period"], "source_url": r["source_url"]}
            for r in state.get("vector_results", [])
        ]

    return {"final_answer": answer, "citations": citations}

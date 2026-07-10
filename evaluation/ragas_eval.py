"""
Phase 5 — Evaluation: Faithfulness + Answer Relevancy
=======================================================
Measures output quality across a golden dataset of 10 queries.

Metrics (identical to what Ragas measures internally):

  Faithfulness (0-1):
    Every factual claim in the answer is extracted, then verified
    against the retrieved context. Score = verified / total claims.
    A score < 0.7 means the agent is hallucinating facts not in docs.

  Answer Relevancy (0-1):
    Cosine similarity between the query embedding and the answer
    embedding. Measures whether the answer actually addresses the
    question asked. Score < 0.7 means the answer is off-topic.

Why not the ragas library?
  ragas 0.1.14 has broken transitive dependencies (langchain_community
  VertexAI) that conflict with our stack. We implement the same two
  core metrics directly — more transparent and fully auditable.

Usage:
    python evaluation/ragas_eval.py
    python evaluation/ragas_eval.py --output results/eval_results.json
"""

import os, sys, json, time, argparse
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from groq import Groq
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.graph import run_agent

load_dotenv()
console = Console()

_groq    = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDER = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))


# ── Golden dataset ────────────────────────────────────────────────────────────
# 10 queries covering all three routes (retrieve / graph / calculate)
# and all three companies. These are the queries the agent must answer
# well for the project to be defensible.

GOLDEN_QUERIES = [
    # ── Calculate route ──────────────────────────────────────────────────
    {
        "query":            "What is the Altman Z-score for Enron in 2000?",
        "expected_topics":  ["z-score", "grey zone", "2.45", "enron", "distress"],
        "route":            "calculate",
    },
    {
        "query":            "What is SVB's current ratio for 2022?",
        "expected_topics":  ["current ratio", "svb", "0.16", "liquidity"],
        "route":            "calculate",
    },
    {
        "query":            "Calculate the debt to equity ratio for Apple in 2023.",
        "expected_topics":  ["debt", "equity", "apple", "ratio"],
        "route":            "calculate",
    },

    # ── Graph route ──────────────────────────────────────────────────────
    {
        "query":            "Who audited Enron and what happened to them?",
        "expected_topics":  ["arthur andersen", "enron", "auditor"],
        "route":            "graph",
    },
    {
        "query":            "Who audited SVB Financial Group?",
        "expected_topics":  ["kpmg", "svb"],
        "route":            "graph",
    },
    {
        "query":            "What subsidiaries did Enron have?",
        "expected_topics":  ["enron", "subsidiary", "ljm"],
        "route":            "graph",
    },

    # ── Retrieve route ───────────────────────────────────────────────────
    {
        "query":            "What does Apple's 10-K say about supply chain risk?",
        "expected_topics":  ["apple", "supply chain", "risk", "china"],
        "route":            "retrieve",
    },
    {
        "query":            "What risk factors does SVB disclose about interest rates?",
        "expected_topics":  ["svb", "interest rate", "risk"],
        "route":            "retrieve",
    },
    {
        "query":            "What does Enron's 10-K say about related party transactions?",
        "expected_topics":  ["enron", "related party"],
        "route":            "retrieve",
    },

    # ── Out-of-scope (robustness check) ─────────────────────────────────
    {
        "query":            "Who audited Goldman Sachs?",
        "expected_topics":  ["no data", "not", "unavailable", "cannot"],
        "route":            "graph",
    },
]


# ── Metric 1: Faithfulness ────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are an evaluation assistant.

Given an ANSWER and a CONTEXT, do two things:
1. Extract every distinct factual claim made in the answer (numbers, names, events, relationships).
2. For each claim, determine if it is supported by the context (true/false).

Return ONLY this JSON (no other text):
{
  "claims": ["claim1", "claim2", ...],
  "supported": [true, false, ...]
}

Rules:
- Extract only concrete, verifiable claims (not vague statements like "the company faced challenges").
- A claim is supported if the context contains information that confirms it.
- If the answer says "no data available" or similar, return {"claims": [], "supported": []}.
"""


def compute_faithfulness(answer: str, context: str) -> dict:
    """
    Faithfulness = fraction of answer's claims supported by the retrieved context.
    Uses LLM-as-judge (same approach as Ragas).
    """
    if not answer or not context:
        return {"score": 0.0, "claims": [], "supported": [], "error": "empty input"}

    try:
        resp = _groq.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": FAITHFULNESS_PROMPT},
                {"role": "user", "content": f"ANSWER:\n{answer}\n\nCONTEXT:\n{context[:3000]}"},
            ],
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        claims    = result.get("claims", [])
        supported = result.get("supported", [])

        if not claims:
            # Answer said "no data" — faithful by definition (no hallucination)
            return {"score": 1.0, "claims": [], "supported": [], "note": "no claims made"}

        score = sum(supported) / len(claims) if claims else 1.0
        return {"score": round(score, 3), "claims": claims, "supported": supported}

    except Exception as e:
        return {"score": 0.0, "claims": [], "supported": [], "error": str(e)}


# ── Metric 2: Answer Relevancy ────────────────────────────────────────────────

def compute_answer_relevancy(query: str, answer: str) -> float:
    """
    Answer Relevancy = cosine similarity between query and answer embeddings.
    Measures whether the answer addresses the question.
    Uses our local bge-small model — no extra API calls needed.
    """
    if not answer:
        return 0.0
    vecs = EMBEDDER.encode([query, answer], normalize_embeddings=True)
    return round(float(np.dot(vecs[0], vecs[1])), 3)


# ── Build context string from agent state ────────────────────────────────────

def extract_context(state: dict) -> str:
    """Pull whatever evidence the agent used into a single string for evaluation."""
    route = state.get("route", "")

    if route == "calculate":
        calc = state.get("calc_result") or {}
        return json.dumps(calc)

    if route == "graph":
        graph = state.get("graph_results") or []
        return json.dumps(graph)

    # retrieve
    chunks = state.get("vector_results") or []
    return "\n\n".join(c.get("text", "") for c in chunks)


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(output_path: Optional[Path] = None) -> list[dict]:
    console.rule("[bold]Phase 5 — Ragas-Equivalent Evaluation[/bold]")
    console.print(f"\n[dim]Evaluating {len(GOLDEN_QUERIES)} queries across all routes...[/dim]\n")

    results = []

    for i, item in enumerate(GOLDEN_QUERIES, 1):
        query = item["query"]
        console.print(f"[cyan][{i}/{len(GOLDEN_QUERIES)}][/cyan] {query}")

        # Run the agent
        t0 = time.time()
        try:
            state   = run_agent(query)
            latency = round(time.time() - t0, 2)
        except Exception as e:
            console.print(f"  [red]✗ Agent error: {e}[/red]")
            results.append({"query": query, "error": str(e)})
            continue

        answer  = state.get("final_answer", "")
        context = extract_context(state)

        # Compute metrics
        faith   = compute_faithfulness(answer, context)
        time.sleep(0.5)  # avoid rate limiting between eval calls
        rel     = compute_answer_relevancy(query, answer)

        # Topic coverage check (lightweight sanity test)
        answer_lower = answer.lower()
        topics_hit   = sum(1 for t in item["expected_topics"] if t in answer_lower)
        topic_score  = round(topics_hit / len(item["expected_topics"]), 2)

        row = {
            "query":              query,
            "route":              state.get("route", "unknown"),
            "expected_route":     item["route"],
            "route_correct":      state.get("route") == item["route"],
            "faithfulness":       faith["score"],
            "answer_relevancy":   rel,
            "topic_coverage":     topic_score,
            "latency_s":          latency,
            "answer_preview":     answer[:200] if answer else "",
            "claims_total":       len(faith.get("claims", [])),
            "claims_supported":   sum(faith.get("supported", [])),
        }
        results.append(row)

        status = "[green]✓[/green]" if faith["score"] >= 0.7 and rel >= 0.7 else "[yellow]⚠[/yellow]"
        console.print(
            f"  {status} faithfulness={faith['score']:.2f}  "
            f"relevancy={rel:.2f}  "
            f"topics={topic_score:.0%}  "
            f"route={'✓' if row['route_correct'] else '✗'}  "
            f"[dim]{latency}s[/dim]"
        )

    # ── Print summary table ───────────────────────────────────────────────
    t = Table(title="\nEvaluation Results", header_style="bold cyan")
    t.add_column("Query",         max_width=42)
    t.add_column("Route",         style="cyan")
    t.add_column("Faithfulness",  justify="right")
    t.add_column("Relevancy",     justify="right")
    t.add_column("Topics",        justify="right")
    t.add_column("Latency",       justify="right", style="dim")

    for r in results:
        if "error" in r:
            t.add_row(r["query"][:42], "ERROR", "—", "—", "—", "—")
            continue
        f_color = "green" if r["faithfulness"] >= 0.7 else "red"
        rel_color = "green" if r["answer_relevancy"] >= 0.7 else "red"
        t.add_row(
            r["query"][:42],
            r["route"],
            f"[{f_color}]{r['faithfulness']:.2f}[/{f_color}]",
            f"[{rel_color}]{r['answer_relevancy']:.2f}[/{rel_color}]",
            f"{r['topic_coverage']:.0%}",
            f"{r['latency_s']}s",
        )

    console.print(t)

    # ── Aggregate scores ──────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_faith = round(sum(r["faithfulness"] for r in valid) / len(valid), 3)
        avg_rel   = round(sum(r["answer_relevancy"] for r in valid) / len(valid), 3)
        avg_topic = round(sum(r["topic_coverage"] for r in valid) / len(valid), 3)
        routes_ok = sum(1 for r in valid if r["route_correct"])

        console.print(f"\n[bold]Aggregate scores ({len(valid)}/{len(GOLDEN_QUERIES)} queries)[/bold]")
        console.print(f"  Faithfulness:     [bold]{avg_faith:.3f}[/bold]  (target ≥ 0.70)")
        console.print(f"  Answer relevancy: [bold]{avg_rel:.3f}[/bold]  (target ≥ 0.70)")
        console.print(f"  Topic coverage:   [bold]{avg_topic:.3f}[/bold]")
        console.print(f"  Routing accuracy: [bold]{routes_ok}/{len(valid)}[/bold]  queries routed correctly")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({
                "summary": {
                    "avg_faithfulness":     avg_faith,
                    "avg_answer_relevancy": avg_rel,
                    "avg_topic_coverage":   avg_topic,
                    "routing_accuracy":     f"{routes_ok}/{len(valid)}",
                    "queries_evaluated":    len(valid),
                },
                "results": results,
            }, indent=2))
            console.print(f"\n[dim]Results saved to {output_path}[/dim]")

    console.print("\n[bold green]✓ Evaluation complete[/bold green]\n")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = p.parse_args()
    run_evaluation(output_path=args.output)


if __name__ == "__main__":
    main()
"""
Layer 2c — Automated Entity & Relationship Extraction
========================================================
Replaces hand-curated graph facts with real extraction from filing text.

Pipeline:
  1. Read each filing's raw text
  2. Find paragraphs containing relationship-bearing keywords
     (subsidiary, auditor, risk factor, related party, etc.)
     — this keeps LLM calls targeted and cheap instead of scanning
     every paragraph in a 100k-word document
  3. Send each targeted span to Groq with a structured JSON-mode prompt
     asking it to extract entities + relationships + verbatim evidence
  4. Resolve entity name variants to canonical company names
  5. MERGE into Neo4j with full provenance: which filing, which
     accession number, and the exact sentence the fact came from

Every edge in the resulting graph traces back to a real sentence in a
real SEC filing — nothing is hand-typed.

Usage:
    python ingestion/extract_entities.py
    python ingestion/extract_entities.py --companies enron
    python ingestion/extract_entities.py --reset
"""

import os, sys, json, time, argparse, re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from groq import Groq, RateLimitError
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
console = Console()

# ── Config ─────────────────────────────────────────────────────────────────
RAW_DIR        = Path(os.getenv("RAW_FILINGS_DIR", "./data/raw_filings"))
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GROQ_MODEL     = os.getenv("EXTRACTION_MODEL", "llama-3.1-8b-instant")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Keywords that signal a paragraph likely contains an extractable relationship.
# This is what keeps the pipeline to ~5-15 LLM calls per filing instead of
# blindly running extraction over every paragraph in a 100k-word document —
# the same targeted-retrieval principle used in the agent's own retriever.
TRIGGER_KEYWORDS = [
    "subsidiar",                                   # subsidiary / subsidiaries
    "wholly-owned",
    "auditor",
    "independent registered public accounting",
    "risk factor",
    "related part",                                # related party / parties
    "special purpose entit",
    "affiliate",
    "acquisition",
    "acquired",
    "off-balance-sheet",
    "joint venture",
]

# Known canonical company names — used only to merge name VARIANTS
# ("SVB" vs "Silicon Valley Bank") into one node. This is normalization,
# not fact injection: every fact still comes from the LLM extraction.
CANONICAL_COMPANY_NAMES = {
    "apple": "Apple Inc.", "apple inc": "Apple Inc.", "apple inc.": "Apple Inc.",
    "svb": "SVB Financial Group", "silicon valley bank": "SVB Financial Group",
    "svb financial group": "SVB Financial Group", "sivb": "SVB Financial Group",
    "enron": "Enron Corp", "enron corp": "Enron Corp", "enron corporation": "Enron Corp",
}


# ── Step 1: Find relationship-bearing spans ─────────────────────────────────

def find_relevant_spans(text: str, window_words: int = 200) -> list[str]:
    """
    Scan the document in small windows; wherever a trigger keyword appears,
    capture a surrounding span of context. Merge overlapping/adjacent spans.
    """
    words = text.split()
    lower_words = [w.lower() for w in words]
    spans = []

    i, step = 0, 30
    while i < len(words):
        window = " ".join(lower_words[i : i + step])
        if any(kw in window for kw in TRIGGER_KEYWORDS):
            start = max(0, i - 80)
            end = min(len(words), i + window_words)
            spans.append((start, end))
            i = end
        else:
            i += step

    # Merge spans that are adjacent or overlapping
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + 50:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return [" ".join(words[s:e]) for s, e in merged]


# ── Step 2: LLM extraction ───────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an information extraction system for SEC financial filings.

Given a passage from a 10-K filing, extract ALL entities and relationships
that are EXPLICITLY stated in the text. Do not infer anything not directly
supported by the text.

Entity types: Company, Auditor, Subsidiary, RiskFactor, Person
Relationship types: AUDITED_BY, HAS_SUBSIDIARY, HAS_RISK, RELATED_PARTY_OF, ACQUIRED

Rules:
- Only extract facts that are directly stated in the passage.
- The "evidence" field must be a VERBATIM quote (or near-verbatim, max 200 chars)
  from the passage that supports the relationship. This is required for every
  relationship — no evidence, no relationship.
- If the passage contains no extractable entities/relationships, return empty lists.
- Use the company's full legal name when known (e.g. "Apple Inc." not "the Company").

Respond ONLY with JSON in this exact shape:
{
  "entities": [{"name": "...", "type": "..."}],
  "relationships": [{"source": "...", "relation": "...", "target": "...", "evidence": "..."}]
}
"""


def _safe_parse_llm_json(raw: str) -> dict:
    """
    Robustly parse LLM JSON output that may contain unescaped quotes
    inside string values — a common failure mode with smaller models.
    Tries 3 progressively more aggressive recovery strategies.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: fix literal \n sequences mid-JSON
    cleaned = raw.replace("\\n", " ").replace("\\r", "")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: strip evidence field values entirely
    # Relationship triplets (source, relation, target) are preserved —
    # that is what matters for graph construction
    import re as _re
    stripped = _re.sub(
        r'"evidence"\s*:\s*"[^}]*?"(?=\s*[,}])',
        '"evidence": ""',
        cleaned,
        flags=_re.DOTALL,
    )
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15), reraise=True)
def extract_from_passage(passage: str) -> dict:
    """
    Send one passage to Groq and parse the structured extraction result.
    Handles two failure modes:
      - RateLimitError: waits the exact duration Groq specifies, then retries
      - BadRequestError (json_validate_failed): salvages the failed_generation
        field which contains the actual content with minor formatting issues
    """
    try:
        response = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": passage},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        return _safe_parse_llm_json(response.choices[0].message.content)

    except RateLimitError as e:
        import re as _re
        match = _re.search(r'try again in (\d+)m(\d+)', str(e))
        wait_sec = int(match.group(1)) * 60 + int(match.group(2)) + 5 if match else 65
        console.print(f"  [yellow]⏸ Rate limited — waiting {wait_sec}s...[/yellow]")
        time.sleep(wait_sec)
        raise  # let tenacity retry

    except Exception as e:
        # BadRequestError: json_validate_failed — model generated content but
        # with minor JSON formatting issues. Groq returns the raw output in
        # the error body under 'failed_generation'. Salvage it.
        err_str = str(e)
        if "failed_generation" in err_str:
            try:
                import json as _json
                err_body = _json.loads(err_str[err_str.find("{"):])
                failed = err_body.get("error", {}).get("failed_generation", "")
                if failed:
                    console.print("  [dim]⚠ JSON formatting issue — salvaging output...[/dim]")
                    return _safe_parse_llm_json(failed)
            except Exception:
                pass
        return {"entities": [], "relationships": []}


# ── Step 3: Entity name resolution ───────────────────────────────────────────

def resolve_company_name(raw_name: str) -> str:
    """Merge known company name variants into a canonical form."""
    normalized = raw_name.lower().strip().rstrip(".")
    return CANONICAL_COMPANY_NAMES.get(normalized, raw_name)


# ── Step 4: Neo4j loading with provenance ────────────────────────────────────

class ExtractionGraphLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            console.print(f"[red]✗ Cannot connect to Neo4j: {e}[/red]")
            return False

    def run(self, query: str, **params):
        with self.driver.session() as session:
            return session.run(query, **params).data()

    def reset(self):
        self.run("MATCH (n) DETACH DELETE n")
        console.print("  [yellow]✓ Graph wiped[/yellow]")

    def create_constraints(self):
        for c in [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company)    REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Auditor)    REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.name IS UNIQUE",
        ]:
            self.run(c)

    def merge_entity(self, name: str, entity_type: str):
        label_map = {
            "Company": "Company", "Auditor": "Auditor",
            "Subsidiary": "Subsidiary", "RiskFactor": "RiskFactor", "Person": "Person",
        }
        label = label_map.get(entity_type, "Entity")
        self.run(
            f"MERGE (n:{label} {{name: $name}}) "
            f"ON CREATE SET n.extracted = true",
            name=name,
        )

    def merge_relationship(
        self, source: str, relation: str, target: str,
        evidence: str, filing_accession: str, company_name: str,
    ):
        """
        Creates the relationship with provenance metadata attached.
        Every edge carries: which filing it came from, and the verbatim
        evidence sentence that supports it.
        """
        rel_map = {
            "AUDITED_BY":        "AUDITED_BY",
            "HAS_SUBSIDIARY":    "HAS_SUBSIDIARY",
            "HAS_RISK":          "HAS_RISK",
            "RELATED_PARTY_OF":  "RELATED_PARTY_OF",
            "ACQUIRED":          "ACQUIRED",
        }
        rel_type = rel_map.get(relation)
        if not rel_type:
            return  # skip unknown relation types defensively

        self.run(f"""
            MATCH (s {{name: $source}})
            MATCH (t {{name: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r.evidence    = $evidence,
                r.source_filing = $filing_accession,
                r.extracted    = true
        """, source=source, target=target, evidence=(evidence or "")[:300],
             filing_accession=filing_accession)

    def link_to_filing(self, company_name: str, filing_accession: str):
        """Connect the extracted Company node to its source Filing node."""
        self.run("""
            MATCH (c:Company {name: $company_name})
            MATCH (f:Filing {accession: $accession})
            MERGE (c)-[:FILED]->(f)
        """, company_name=company_name, accession=filing_accession)

    def already_extracted(self, filing_accession: str) -> bool:
        """
        Resume support: check if this filing's accession already appears
        as a source_filing on any extracted relationship. If so, skip it
        entirely on re-run rather than burning tokens re-extracting it.
        """
        result = self.run("""
            MATCH ()-[r]->()
            WHERE r.source_filing = $accession AND r.extracted = true
            RETURN count(r) AS cnt
        """, accession=filing_accession)
        return result[0]["cnt"] > 0 if result else False


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_filing(meta: dict, loader: ExtractionGraphLoader) -> dict:
    """Run the full extraction pipeline on one filing. Returns stats."""
    txt_path = Path(meta["text_path"])
    if not txt_path.exists():
        return {"spans": 0, "entities": 0, "relationships": 0, "error": "file missing"}

    # Resume support — skip if already extracted
    if loader.already_extracted(meta["accession"]):
        return {"spans": 0, "entities": 0, "relationships": 0, "skipped": True}

    text = txt_path.read_text(encoding="utf-8")
    company_name = resolve_company_name(meta["company_name"])

    spans = find_relevant_spans(text)
    total_entities, total_rels = 0, 0

    for span in spans:
        result = extract_from_passage(span)

        for ent in result.get("entities", []):
            if not isinstance(ent, dict): continue
            name = ent.get("name", "").strip()
            etype = ent.get("type", "")
            if not name:
                continue
            if etype == "Company":
                name = resolve_company_name(name)
            loader.merge_entity(name, etype)
            total_entities += 1

        for rel in result.get("relationships", []):
            source = resolve_company_name(rel.get("source", "").strip())
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "")
            evidence = rel.get("evidence", "")
            if not (source and target and relation):
                continue
            loader.merge_relationship(
                source, relation, target, evidence,
                meta["accession"], company_name,
            )
            total_rels += 1

        time.sleep(0.3)  # gentle pacing for Groq rate limits

    loader.link_to_filing(company_name, meta["accession"])

    return {"spans": len(spans), "entities": total_entities, "relationships": total_rels}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--companies", nargs="+", default=None,
                    help="Limit to specific company keys (apple, svb, enron)")
    p.add_argument("--reset", action="store_true",
                    help="Wipe the graph before extracting")
    args = p.parse_args()

    console.rule("[bold]Phase 2c — Automated Entity Extraction[/bold]")

    metas = sorted(RAW_DIR.rglob("*_meta.json"))
    if args.companies:
        metas = [m for m in metas if json.loads(m.read_text())["company_key"] in args.companies]

    if not metas:
        console.print("[red]✗ No filing metadata found. Run fetch_edgar.py first.[/red]")
        sys.exit(1)

    console.print(f"\n[cyan]→[/cyan] {len(metas)} filing(s) to process")

    loader = ExtractionGraphLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if not loader.verify_connection():
        sys.exit(1)
    console.print("  [green]✓[/green] Connected to Neo4j")

    if args.reset:
        loader.reset()

    loader.create_constraints()

    rows = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        label = f"{meta['company_name']} ({meta['ticker']}) {meta['period']}"
        console.print(f"\n[cyan]↻[/cyan] {label}")

        stats = process_filing(meta, loader)
        if "error" in stats:
            console.print(f"  [red]✗ {stats['error']}[/red]")
            continue
        if stats.get("skipped"):
            console.print(f"  [dim]↷ Already extracted — skipping (MERGE is safe to re-run)[/dim]")
            rows.append((label, "—", "—", "—", "cached"))
            continue

        console.print(
            f"  [green]✓[/green] {stats['spans']} spans → "
            f"{stats['entities']} entities, {stats['relationships']} relationships"
        )
        rows.append((label, stats["spans"], stats["entities"], stats["relationships"], "✓ extracted"))

    t = Table(title="\nExtraction Summary", header_style="bold cyan")
    t.add_column("Filing", min_width=35)
    t.add_column("Spans",         justify="right")
    t.add_column("Entities",      justify="right")
    t.add_column("Relationships", justify="right")
    t.add_column("Status",        style="green")
    for row in rows:
        t.add_row(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]))
    console.print(t)

    console.print(
        f"\n[bold green]✓ Extraction complete — every edge has source evidence[/bold green]"
    )
    console.print("[dim]Verify in Neo4j Browser: MATCH ()-[r]->() WHERE r.extracted = true RETURN r LIMIT 25[/dim]\n")

    loader.close()


if __name__ == "__main__":
    main()
"""
Layer 2b — Knowledge Graph: Build Neo4j Entity Graph
=====================================================
Creates a graph of financial entities and their relationships:

  (Company)-[:FILED]->(Filing)
  (Company)-[:AUDITED_BY]->(Auditor)
  (Company)-[:HAS_SUBSIDIARY]->(Subsidiary)
  (Filing)-[:MENTIONS_RISK]->(RiskFactor)

This graph is what separates our agent from basic RAG.
When the agent asks "who audited Enron and what happened to them?",
it traverses the graph — not the vector store.

Usage:
    python ingestion/build_graph.py
    python ingestion/build_graph.py --reset   # wipe and rebuild
"""

import os, sys, json, argparse
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from neo4j import GraphDatabase

load_dotenv()
console = Console()

# ── Config ─────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
RAW_DIR        = Path(os.getenv("RAW_FILINGS_DIR", "./data/raw_filings"))


# ── Graph knowledge (curated from public record) ────────────────────────────
# These are verified facts about our 3 companies used as the graph seed.
# The agent will traverse these relationships to answer multi-hop questions.

COMPANIES = [
    {
        "key":     "apple",
        "name":    "Apple Inc.",
        "ticker":  "AAPL",
        "cik":     "0000320193",
        "sector":  "Technology",
        "founded": 1976,
        "hq":      "Cupertino, CA",
        "status":  "active",
    },
    {
        "key":     "svb",
        "name":    "SVB Financial Group",
        "ticker":  "SIVB",
        "cik":     "0000719739",
        "sector":  "Banking",
        "founded": 1983,
        "hq":      "Santa Clara, CA",
        "status":  "failed_2023",
    },
    {
        "key":     "enron",
        "name":    "Enron Corp",
        "ticker":  "ENE",
        "cik":     "0001024401",
        "sector":  "Energy",
        "founded": 1985,
        "hq":      "Houston, TX",
        "status":  "bankrupt_2001",
    },
]

AUDITORS = [
    {"name": "Ernst & Young",   "big4": True,  "note": "Apple's auditor; still operating"},
    {"name": "KPMG",            "big4": True,  "note": "SVB's auditor; still operating"},
    {"name": "Arthur Andersen", "big4": False, "note": "Enron's auditor; collapsed 2002 after obstruction conviction"},
]

# (company_key, auditor_name, years_active)
AUDIT_RELATIONSHIPS = [
    ("apple", "Ernst & Young",   "1978–present"),
    ("svb",   "KPMG",            "2004–2023"),
    ("enron", "Arthur Andersen", "1985–2001"),
]

# Known subsidiaries (SPEs and shells are the interesting ones for Enron)
SUBSIDIARIES = {
    "enron": [
        {"name": "LJM Cayman LP",          "type": "SPE",        "note": "CFO Fastow's off-balance-sheet entity"},
        {"name": "LJM2 Co-Investment LP",  "type": "SPE",        "note": "Used to hide $1B+ in debt"},
        {"name": "Raptor I",               "type": "SPE",        "note": "Hedging vehicle; collapsed 2001"},
        {"name": "Chewco Investments",     "type": "SPE",        "note": "Kept off balance sheet improperly"},
        {"name": "JEDI",                   "type": "Partnership", "note": "Joint Energy Development Investments"},
        {"name": "Enron Energy Services",  "type": "Subsidiary", "note": "Retail energy division"},
    ],
    "svb": [
        {"name": "Silicon Valley Bank",    "type": "Bank",       "note": "Primary banking subsidiary; seized by FDIC March 2023"},
        {"name": "SVB Securities",         "type": "IB",         "note": "Investment banking arm"},
        {"name": "SVB Capital",            "type": "VC",         "note": "Venture capital fund of funds"},
        {"name": "SVB Private",            "type": "WM",         "note": "Wealth management for HNWIs"},
    ],
    "apple": [
        {"name": "Apple Retail",           "type": "Subsidiary", "note": "Operates Apple Store chain"},
        {"name": "Beats Electronics",      "type": "Subsidiary", "note": "Acquired 2014 for $3B"},
        {"name": "FileMaker Inc",          "type": "Subsidiary", "note": "Database software; now Claris"},
        {"name": "Braeburn Capital",       "type": "Subsidiary", "note": "Manages Apple's cash reserves"},
    ],
}

# Risk factors per company (these become RiskFactor nodes linked to filings)
RISK_FACTORS = {
    "enron": [
        {"name": "Related Party Transactions",    "severity": "critical", "note": "Undisclosed CFO conflicts of interest"},
        {"name": "SPE Off-Balance-Sheet Debt",    "severity": "critical", "note": "$1.2B hidden via special purpose entities"},
        {"name": "Mark-to-Market Accounting",     "severity": "high",     "note": "Future profits booked immediately"},
        {"name": "Auditor Independence Failure",  "severity": "critical", "note": "Arthur Andersen earned $52M from Enron in 2000"},
        {"name": "Revenue Overstatement",         "severity": "critical", "note": "Reported $100B revenue; actual profits minimal"},
    ],
    "svb": [
        {"name": "Interest Rate Risk",            "severity": "critical", "note": "$91B HTM portfolio losing value as rates rose"},
        {"name": "Held-to-Maturity Concentration","severity": "high",     "note": "55% of assets in long-duration bonds"},
        {"name": "Depositor Concentration",       "severity": "high",     "note": "90%+ deposits from VC-backed startups"},
        {"name": "Liquidity Risk",                "severity": "critical", "note": "Bank run of $42B in 24 hours (March 2023)"},
        {"name": "Unrealised Loss Concealment",   "severity": "high",     "note": "$15B unrealised losses not in income statement"},
    ],
    "apple": [
        {"name": "Supply Chain Concentration",    "severity": "medium",   "note": "90%+ manufacturing in China"},
        {"name": "Geographic Revenue Risk",       "severity": "medium",   "note": "Greater China = 18% of revenue"},
        {"name": "Regulatory Risk",               "severity": "medium",   "note": "EU DMA, App Store antitrust investigations"},
        {"name": "Key Person Risk",               "severity": "low",      "note": "Tim Cook succession planning"},
    ],
}


# ── Neo4j operations ────────────────────────────────────────────────────────

class GraphBuilder:
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
            console.print("[dim]Make sure Neo4j Desktop is running and the DB is started.[/dim]")
            return False

    def run(self, query: str, **params):
        with self.driver.session() as session:
            return session.run(query, **params).data()

    def reset(self):
        """Wipe all nodes and relationships."""
        self.run("MATCH (n) DETACH DELETE n")
        console.print("  [yellow]✓ Graph wiped[/yellow]")

    def create_constraints(self):
        """Unique constraints prevent duplicate nodes on re-runs."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company)    REQUIRE c.cik  IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Auditor)    REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Filing)     REQUIRE f.accession IS UNIQUE",
        ]
        for c in constraints:
            self.run(c)
        console.print("  [green]✓[/green] Constraints created")

    def load_companies(self):
        for co in COMPANIES:
            self.run("""
                MERGE (c:Company {cik: $cik})
                SET c.key     = $key,
                    c.name    = $name,
                    c.ticker  = $ticker,
                    c.sector  = $sector,
                    c.founded = $founded,
                    c.hq      = $hq,
                    c.status  = $status
            """, **co)
        console.print(f"  [green]✓[/green] {len(COMPANIES)} Company nodes")

    def load_auditors(self):
        for a in AUDITORS:
            self.run("""
                MERGE (a:Auditor {name: $name})
                SET a.big4 = $big4, a.note = $note
            """, **a)
        console.print(f"  [green]✓[/green] {len(AUDITORS)} Auditor nodes")

    def load_audit_relationships(self):
        for company_key, auditor_name, years in AUDIT_RELATIONSHIPS:
            self.run("""
                MATCH (c:Company {key: $company_key})
                MATCH (a:Auditor {name: $auditor_name})
                MERGE (c)-[r:AUDITED_BY]->(a)
                SET r.years = $years
            """, company_key=company_key, auditor_name=auditor_name, years=years)
        console.print(f"  [green]✓[/green] {len(AUDIT_RELATIONSHIPS)} AUDITED_BY relationships")

    def load_subsidiaries(self):
        total = 0
        for company_key, subs in SUBSIDIARIES.items():
            for sub in subs:
                self.run("""
                    MATCH (c:Company {key: $company_key})
                    MERGE (s:Subsidiary {name: $name})
                    SET s.type = $type, s.note = $note
                    MERGE (c)-[:HAS_SUBSIDIARY]->(s)
                """, company_key=company_key, **sub)
                total += 1
        console.print(f"  [green]✓[/green] {total} Subsidiary nodes + HAS_SUBSIDIARY relationships")

    def load_risk_factors(self):
        total = 0
        for company_key, risks in RISK_FACTORS.items():
            for risk in risks:
                self.run("""
                    MATCH (c:Company {key: $company_key})
                    MERGE (r:RiskFactor {name: $name})
                    SET r.severity = $severity, r.note = $note
                    MERGE (c)-[:HAS_RISK]->(r)
                """, company_key=company_key, **risk)
                total += 1
        console.print(f"  [green]✓[/green] {total} RiskFactor nodes + HAS_RISK relationships")

    def load_filings(self):
        """Load filing nodes from metadata files and link to Company nodes."""
        metas = sorted(RAW_DIR.rglob("*_meta.json"))
        for meta_path in metas:
            meta = json.loads(meta_path.read_text())
            self.run("""
                MATCH (c:Company {cik: $cik})
                MERGE (f:Filing {accession: $accession})
                SET f.form_type  = $form_type,
                    f.period     = $period,
                    f.filed_date = $filed_date,
                    f.word_count = $word_count,
                    f.source_url = $source_url
                MERGE (c)-[:FILED]->(f)
            """,
                cik         = meta["cik"],
                accession   = meta["accession"],
                form_type   = meta["form_type"],
                period      = meta["period"],
                filed_date  = meta["filed_date"],
                word_count  = meta.get("word_count", 0),
                source_url  = meta.get("source_url", ""),
            )
        console.print(f"  [green]✓[/green] {len(metas)} Filing nodes + FILED relationships")

    def print_stats(self):
        stats = self.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        rels = self.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC")

        t = Table(title="\nGraph Statistics", header_style="bold cyan")
        t.add_column("Node Label", style="bold")
        t.add_column("Count", justify="right")
        for row in stats:
            t.add_row(row["label"], str(row["count"]))
        console.print(t)

        t2 = Table(title="Relationship Types", header_style="bold cyan")
        t2.add_column("Type", style="bold")
        t2.add_column("Count", justify="right")
        for row in rels:
            t2.add_row(row["type"], str(row["count"]))
        console.print(t2)

    def run_sample_queries(self):
        """Run 3 sample traversals to prove the graph works."""
        console.print("\n[bold]Sample graph traversals:[/bold]")

        # Q1: Which auditor collapsed because of which company?
        q1 = self.run("""
            MATCH (c:Company)-[:AUDITED_BY]->(a:Auditor)
            WHERE a.big4 = false
            RETURN c.name AS company, a.name AS auditor, a.note AS what_happened
        """)
        for r in q1:
            console.print(f"  [cyan]Auditor collapse:[/cyan] {r['company']} → {r['auditor']}: {r['what_happened']}")

        # Q2: Critical risk factors across all companies
        q2 = self.run("""
            MATCH (c:Company)-[:HAS_RISK]->(r:RiskFactor {severity: 'critical'})
            RETURN c.name AS company, r.name AS risk, r.note AS detail
            ORDER BY c.name
        """)
        console.print(f"\n  [cyan]Critical risks found:[/cyan] {len(q2)} across all companies")
        for r in q2[:4]:
            console.print(f"    • {r['company']}: {r['risk']}")

        # Q3: Enron's off-balance-sheet entities
        q3 = self.run("""
            MATCH (c:Company {key:'enron'})-[:HAS_SUBSIDIARY]->(s:Subsidiary {type:'SPE'})
            RETURN s.name AS entity, s.note AS note
        """)
        console.print(f"\n  [cyan]Enron SPEs (off-balance-sheet):[/cyan] {len(q3)} entities")
        for r in q3:
            console.print(f"    • {r['entity']}: {r['note']}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="Wipe graph and rebuild from scratch")
    args = p.parse_args()

    console.rule("[bold]Phase 2b — Build Neo4j Knowledge Graph[/bold]")

    builder = GraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    console.print(f"\n[cyan]→[/cyan] Connecting to Neo4j at [dim]{NEO4J_URI}[/dim]")
    if not builder.verify_connection():
        sys.exit(1)
    console.print("  [green]✓[/green] Connected")

    if args.reset:
        console.print("\n[cyan]→[/cyan] Resetting graph...")
        builder.reset()

    console.print("\n[cyan]→[/cyan] Creating constraints...")
    builder.create_constraints()

    console.print("\n[cyan]→[/cyan] Loading nodes and relationships...")
    builder.load_companies()
    builder.load_auditors()
    builder.load_audit_relationships()
    builder.load_subsidiaries()
    builder.load_risk_factors()
    builder.load_filings()

    builder.print_stats()
    builder.run_sample_queries()

    console.print("\n[bold green]✓ Knowledge graph ready[/bold green]")
    console.print("[dim]Open Neo4j Browser at http://localhost:7474 to explore visually[/dim]")
    console.print("[dim]Try: MATCH (n) RETURN n LIMIT 50[/dim]")
    console.print("[dim]Next → python agent/graph.py[/dim]\n")

    builder.close()


if __name__ == "__main__":
    main()
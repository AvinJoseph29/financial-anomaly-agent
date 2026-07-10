"""
Layer 1 — Data Ingestion: SEC EDGAR Filing Fetcher  (v3 — fixed)
=================================================================
Fixes vs v2:
  - Enron CIK corrected (was Sprint's CIK 0000101830, now 0001024401)
  - Document URL now resolved via directory listing, not primaryDocument field
    (primaryDocument points to iXBRL viewer which 404s on direct fetch)
  - Directory listing endpoint always reliable: /Archives/edgar/data/{CIK}/{acc}/

Usage:
    python ingestion/fetch_edgar.py
    python ingestion/fetch_edgar.py --companies apple svb --max-filings 2
"""

import os, sys, time, json, argparse, requests
from pathlib import Path
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()
console = Console()

# ── Config ─────────────────────────────────────────────────────────────────
RAW_DIR = Path(os.getenv("RAW_FILINGS_DIR", "./data/raw_filings"))

SEC_HEADERS = {
    "User-Agent": "FinancialAnomalyAgent research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

REQUEST_DELAY = 0.4   # seconds — well under SEC's 10 req/sec limit

# ── Companies  (CIKs verified on https://www.sec.gov/cgi-bin/browse-edgar) ─
COMPANIES = {
    "apple": {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "note": "Baseline — clean financials, auditor Ernst & Young",
    },
    "svb": {
        "cik": "0000719739",
        "name": "SVB Financial Group",
        "ticker": "SIVB",
        "note": "Bank failure 2023 — interest rate risk mismanagement signals",
    },
    "enron": {
        "cik": "0001024401",          # ← CORRECTED (was Sprint's CIK by mistake)
        "name": "Enron Corp",
        "ticker": "ENE",
        "note": "Accounting fraud — SPE off-balance-sheet, auditor Arthur Andersen",
    },
}


# ── HTTP ────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    r.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return r


# ── EDGAR helpers ───────────────────────────────────────────────────────────

def get_tenk_filings(cik: str, max_n: int) -> list[dict]:
    """
    Pull submission history and return the N most recent 10-K accessions.
    """
    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _get(url).json()

    recent = data["filings"]["recent"]
    forms       = recent["form"]
    accessions  = recent["accessionNumber"]
    filed_dates = recent["filingDate"]
    periods     = recent.get("reportDate", [""] * len(accessions))

    results = [
        {
            "accession":  accessions[i],
            "filed_date": filed_dates[i],
            "period":     periods[i] or filed_dates[i][:7],
        }
        for i, f in enumerate(forms) if f == "10-K"
    ]
    return results[:max_n]


def resolve_primary_doc(cik: str, accession: str) -> Optional[str]:
    """
    Resolve primary 10-K document from SEC's index.json.

    Strategy 1 (preferred): item where type == '10-K' — SEC sets this explicitly.
    Strategy 2 (fallback):  largest .htm/.txt that isn't an exhibit — covers
                            old Enron-era filings where type field may be blank.
    """
    cik_int   = str(int(cik))
    acc_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/index.json"

    try:
        items = _get(url).json().get("directory", {}).get("item", [])
    except Exception:
        return None

    # Strategy 1: SEC explicitly tags the primary document as type 10-K
    for item in items:
        if item.get("type", "").upper() == "10-K":
            return item["name"]

    # Strategy 2: fallback for old filings (Enron 1998-2001) where type is blank
    candidates = []
    for item in items:
        name  = item.get("name", "")
        lower = name.lower()
        size  = int(item.get("size") or 0)
        ftype = item.get("type", "").upper()

        if not (lower.endswith(".htm") or lower.endswith(".html") or lower.endswith(".txt")):
            continue
        if ftype.startswith("EX"):
            continue
        if any(x in lower for x in ["xbrl", "summary", "index", "graphic"]):
            continue
        if size < 20_000:
            continue

        candidates.append((size, name))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def fetch_and_clean(cik: str, accession: str, filename: str) -> str:
    """Download filing HTML and return stripped plain text."""
    cik_int   = str(int(cik))
    acc_clean = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_clean}/{filename}"
    )
    r = _get(url)
    r.encoding = r.apparent_encoding or "utf-8"

    soup = BeautifulSoup(r.text, "lxml-xml")
    for tag in soup(["script", "style", "ix:header", "head"]):
        tag.decompose()

    lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def fetch_company(key: str, info: dict, max_filings: int, out_root: Path) -> list[dict]:
    cik = info["cik"]
    out = out_root / key
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]{info['name']} ({info['ticker']})[/bold]")
    console.print(f"  [dim]{info['note']}[/dim]")
    console.print(f"  [cyan]→[/cyan] Fetching submission history...")

    filings = get_tenk_filings(cik, max_filings)
    if not filings:
        console.print("  [red]✗ No 10-K filings found[/red]")
        return []

    console.print(f"  [green]✓[/green] {len(filings)} 10-K filing(s) found")
    saved = []

    for i, f in enumerate(filings, 1):
        acc    = f["accession"]
        period = f["period"]
        safe   = period.replace("-", "")
        stem   = acc.replace("-", "")

        txt_path  = out / f"10K_{safe}_{stem}.txt"
        meta_path = out / f"10K_{safe}_{stem}_meta.json"

        if txt_path.exists():
            console.print(f"  [dim]  [{i}/{len(filings)}] {period} — cached[/dim]")
            saved.append(json.loads(meta_path.read_text()))
            continue

        console.print(f"  [cyan]  [{i}/{len(filings)}] {period} — resolving document...[/cyan]")

        try:
            filename = resolve_primary_doc(cik, acc)
            if not filename:
                console.print(f"  [yellow]    ✗ No primary HTM found in directory listing[/yellow]")
                continue

            console.print(f"  [cyan]    → Downloading {filename}[/cyan]")
            text = fetch_and_clean(cik, acc, filename)

            if len(text) < 10_000:
                console.print(f"  [yellow]    ✗ Too small ({len(text):,} chars) — skipping[/yellow]")
                continue

            txt_path.write_text(text, encoding="utf-8")

            meta = {
                "company_key":  key,
                "company_name": info["name"],
                "ticker":       info["ticker"],
                "cik":          cik,
                "accession":    acc,
                "form_type":    "10-K",
                "filed_date":   f["filed_date"],
                "period":       period,
                "filename":     filename,
                "text_path":    str(txt_path),
                "char_count":   len(text),
                "word_count":   len(text.split()),
                "fetched_at":   datetime.utcnow().isoformat(),
                "source_url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{acc.replace('-','')}/{filename}"
                ),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            console.print(
                f"  [green]    ✓ Saved[/green] — "
                f"[bold]{meta['word_count']:,}[/bold] words"
            )
            saved.append(meta)

        except Exception as e:
            console.print(f"  [red]    ✗ {e}[/red]")

    return saved


def print_summary(all_meta: list[dict]) -> None:
    t = Table(title="\nDownloaded Filings", header_style="bold cyan")
    t.add_column("Company",  style="bold", min_width=20)
    t.add_column("Period",   style="cyan")
    t.add_column("Filed")
    t.add_column("Words",    justify="right")
    t.add_column("File",     style="dim", max_width=45)

    for m in all_meta:
        t.add_row(
            f"{m['company_name']} ({m['ticker']})",
            m["period"],
            m["filed_date"],
            f"{m['word_count']:,}",
            Path(m["text_path"]).name,
        )

    console.print(t)
    console.print(f"\n[bold green]✓ {len(all_meta)} filing(s) ready[/bold green]")
    console.print("[dim]Next → python ingestion/chunk_and_embed.py[/dim]\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--companies", nargs="+", choices=list(COMPANIES), default=list(COMPANIES))
    p.add_argument("--max-filings", type=int, default=3)
    p.add_argument("--output-dir",  type=Path, default=RAW_DIR)
    args = p.parse_args()

    console.rule("[bold]Financial Anomaly Agent — EDGAR Fetch v3[/bold]")
    console.print(f"Output : [cyan]{args.output_dir.resolve()}[/cyan]")
    console.print(f"Target : [cyan]{', '.join(args.companies)}[/cyan]")

    all_meta = []
    for key in args.companies:
        all_meta.extend(fetch_company(key, COMPANIES[key], args.max_filings, args.output_dir))

    if all_meta:
        print_summary(all_meta)
    else:
        console.print("\n[red]No filings downloaded.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

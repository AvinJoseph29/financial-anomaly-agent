"""
Layer 2 — Chunking, Embedding & Vector Store Loading
=====================================================
Reads all downloaded 10-K .txt files, splits into overlapping chunks,
embeds with BAAI/bge-small-en-v1.5 (local, no API key), and loads
into Qdrant running in local file mode (no Docker, no server).

Usage:
    python ingestion/chunk_and_embed.py
    python ingestion/chunk_and_embed.py --reset   # wipe and rebuild from scratch
"""

import os, sys, json, uuid, argparse
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, Filter, FieldCondition, MatchValue,
)

load_dotenv()
console = Console()

# ── Config ─────────────────────────────────────────────────────────────────
RAW_DIR      = Path(os.getenv("RAW_FILINGS_DIR", "./data/raw_filings"))
QDRANT_PATH  = Path(os.getenv("QDRANT_PATH",     "./data/qdrant_store"))
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL",       "BAAI/bge-small-en-v1.5")

COLLECTION   = "sec_filings"
VECTOR_SIZE  = 384          # bge-small-en-v1.5 output dimension
CHUNK_WORDS  = 512          # words per chunk (≈ 400 tokens, safely under 512)
OVERLAP      = 50           # word overlap between consecutive chunks
BATCH_SIZE   = 64           # chunks per embedding batch


# ── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Word-based (not token-based) because it's deterministic and fast.
    At ~0.75 tokens/word, 512 words ≈ 384 tokens — well inside bge's 512-token limit.
    """
    words = text.split()
    if not words:
        return []

    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap

    return chunks


# ── Qdrant setup ────────────────────────────────────────────────────────────

def get_qdrant_client(path: Path, reset: bool = False) -> QdrantClient:
    """
    Returns a Qdrant client in local file mode (no Docker, no server).
    Data persists on disk at QDRANT_PATH between runs.
    """
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))

    existing = {c.name for c in client.get_collections().collections}

    if reset and COLLECTION in existing:
        client.delete_collection(COLLECTION)
        console.print(f"  [yellow]✓ Collection '{COLLECTION}' reset[/yellow]")
        existing.discard(COLLECTION)

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        # Create payload indices for fast metadata filtering
        client.create_payload_index(COLLECTION, "company_key",  PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "ticker",       PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "period",       PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "form_type",    PayloadSchemaType.KEYWORD)
        console.print(f"  [green]✓ Collection '{COLLECTION}' created (dim={VECTOR_SIZE})[/green]")
    else:
        count = client.count(COLLECTION).count
        console.print(f"  [dim]✓ Collection '{COLLECTION}' exists ({count:,} vectors)[/dim]")

    return client


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_all_metadata() -> list[dict]:
    """Find all *_meta.json files under RAW_DIR."""
    metas = sorted(RAW_DIR.rglob("*_meta.json"))
    if not metas:
        console.print(f"[red]✗ No metadata files found under {RAW_DIR}[/red]")
        console.print("[dim]Run ingestion/fetch_edgar.py first.[/dim]")
        sys.exit(1)
    return [json.loads(m.read_text()) for m in metas]


def already_indexed(client: QdrantClient, accession: str) -> bool:
    """
    Check if this filing's chunks are already in Qdrant (resume support).
    Uses a scroll filter on the accession payload field.
    """
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="accession", match=MatchValue(value=accession))]),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0


def embed_and_upsert(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[str],
    meta: dict,
) -> int:
    """
    Embed a list of chunks in batches and upsert into Qdrant.
    Returns number of points upserted.
    """
    points = []

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        vectors = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine similarity = dot product on normalised vecs
        )

        for i, (chunk_text, vector) in enumerate(zip(batch, vectors)):
            chunk_idx = batch_start + i
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        # ── Text content (what the agent retrieves) ──────────
                        "text":         chunk_text,
                        "chunk_index":  chunk_idx,
                        "chunk_total":  len(chunks),

                        # ── Filing metadata (for filtering + citations) ──────
                        "company_key":  meta["company_key"],
                        "company_name": meta["company_name"],
                        "ticker":       meta["ticker"],
                        "cik":          meta["cik"],
                        "accession":    meta["accession"],
                        "form_type":    meta["form_type"],
                        "filed_date":   meta["filed_date"],
                        "period":       meta["period"],
                        "source_url":   meta.get("source_url", ""),
                    },
                )
            )

    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def run(reset: bool = False) -> None:
    console.rule("[bold]Phase 2 — Chunk, Embed & Load into Qdrant[/bold]")

    # ── Step 1: Load metadata ─────────────────────────────────────────────
    console.print("\n[cyan]→[/cyan] Loading filing metadata...")
    all_meta = load_all_metadata()
    console.print(f"  [green]✓[/green] {len(all_meta)} filing(s) found")

    # ── Step 2: Init Qdrant ───────────────────────────────────────────────
    console.print(f"\n[cyan]→[/cyan] Initialising Qdrant at [dim]{QDRANT_PATH}[/dim]")
    client = get_qdrant_client(QDRANT_PATH, reset=reset)

    # ── Step 3: Load embedding model ──────────────────────────────────────
    console.print(f"\n[cyan]→[/cyan] Loading embedding model [bold]{EMBED_MODEL}[/bold]")
    console.print("  [dim](First run downloads ~33MB — cached afterwards)[/dim]")
    model = SentenceTransformer(EMBED_MODEL)
    console.print("  [green]✓[/green] Model ready")

    # ── Step 4: Process each filing ───────────────────────────────────────
    console.print("\n[cyan]→[/cyan] Processing filings...\n")

    summary_rows = []
    total_chunks = 0

    for meta in all_meta:
        label = f"{meta['company_name']} ({meta['ticker']}) {meta['period']}"

        # Resume: skip if already indexed
        if already_indexed(client, meta["accession"]):
            console.print(f"  [dim]↷  {label} — already in Qdrant, skipping[/dim]")
            summary_rows.append((label, meta["word_count"], "—", "cached"))
            continue

        txt_path = Path(meta["text_path"])
        if not txt_path.exists():
            console.print(f"  [red]✗  {label} — text file missing: {txt_path}[/red]")
            continue

        text   = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        console.print(
            f"  [cyan]↻[/cyan]  {label} — "
            f"{meta['word_count']:,} words → [bold]{len(chunks)}[/bold] chunks"
        )

        n = embed_and_upsert(client, model, chunks, meta)
        total_chunks += n

        console.print(f"  [green]✓[/green]  Upserted {n:,} vectors")
        summary_rows.append((label, meta["word_count"], len(chunks), "✓ indexed"))

    # ── Step 5: Summary ───────────────────────────────────────────────────
    total_vectors = client.count(COLLECTION).count

    t = Table(title="\nIngestion Summary", header_style="bold cyan")
    t.add_column("Filing",   min_width=38)
    t.add_column("Words",    justify="right")
    t.add_column("Chunks",   justify="right")
    t.add_column("Status",   style="green")

    for label, words, chunks, status in summary_rows:
        t.add_row(label, f"{words:,}" if isinstance(words, int) else words,
                  f"{chunks:,}" if isinstance(chunks, int) else chunks, status)

    console.print(t)
    console.print(f"\n[bold green]✓ Qdrant collection '{COLLECTION}': {total_vectors:,} total vectors[/bold green]")
    console.print(f"[dim]Store path: {QDRANT_PATH.resolve()}[/dim]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true",
                   help="Delete and rebuild the Qdrant collection from scratch")
    args = p.parse_args()
    run(reset=args.reset)


if __name__ == "__main__":
    main()

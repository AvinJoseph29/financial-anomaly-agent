# Financial Anomaly Investigation Agent

A production-grade agentic RAG system that investigates financial anomalies in SEC 10-K filings. Built with LangGraph, Qdrant, Neo4j, FastAPI, and a React UI with live knowledge graph visualization.

> **Stack:** Python 3.9 · LangGraph · Qdrant · Neo4j · Groq (Llama 3.3 70B) · FastAPI · React + Vite · D3.js

---

## Demo



### Demo Video
https://github.com/user-attachments/assets/1a5cbf19-1641-478c-9f39-d04d6136cb88


### Screenshots

| Chat Interface | Knowledge Graph Evidence |
|---|---|
| ![Chat UI](docs/screenshots/chat_ui.png) | ![Graph Viz](docs/screenshots/graph_viz.png) |

| Neo4j Graph Explorer | Evaluation Results |
|---|---|
| ![Neo4j](docs/screenshots/neo4j_graph.png) | ![Eval](docs/screenshots/eval_results.png) |

---

## What It Does

Given a natural language query about a public company, the agent:

1. **Plans** — decides whether the query needs document retrieval, graph traversal, or financial calculation
2. **Retrieves** — searches 1,385 vector embeddings across 9 real SEC filings (Qdrant), or traverses entity relationships in a knowledge graph (Neo4j)
3. **Calculates** — computes financial ratios (Altman Z-score, current ratio, debt/equity) using verified figures extracted from filings
4. **Synthesises** — writes a structured investigation memo with citations traceable to specific SEC filings

Every graph edge carries a verbatim evidence sentence from the source filing. Every vector retrieval links back to the original SEC EDGAR document.

### Example Queries

```
What is the Altman Z-score for Enron in 2000?
→ Z-score: 2.45 (Grey Zone) — looked survivable on paper weeks before collapse

Who audited SVB Financial Group?
→ KPMG, 2004–2023. Graph shows auditor → company relationship with years active.

What does SVB's 10-K say about interest rate risk?
→ Retrieves relevant chunks from 3 years of SVB filings with SEC citations.

What subsidiaries did Enron have?
→ Renders force-directed graph: Enron → LJM Cayman LP, Raptor I, Chewco Investments (SPEs)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         React UI (Vite)                         │
│  Sidebar · Chat thread · D3 Knowledge Graph · Citations panel   │
└────────────────────────┬────────────────────────────────────────┘
                         │  SSE streaming  /investigate/stream
┌────────────────────────▼────────────────────────────────────────┐
│                    FastAPI  (api/main.py)                        │
│              Streaming · REST · graph_data builder              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              LangGraph Agent  (agent/graph.py)                  │
│                                                                 │
│   ┌──────────┐    ┌───────────┐    ┌────────────┐              │
│   │ Planner  │───▶│ Retriever │───▶│ Responder  │              │
│   │  (LLM)   │    │           │    │   (LLM)    │              │
│   └──────────┘    └─────┬─────┘    └────────────┘              │
│         │               │                                        │
│         │ calculate     ├── Vector search  →  Qdrant            │
│         ▼               └── Graph traversal →  Neo4j            │
│   ┌──────────┐                                                   │
│   │Calculator│  Altman Z · Current Ratio · Debt/Equity          │
│   └──────────┘                                                   │
└─────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      Data Layer                                  │
│                                                                 │
│  Qdrant (local)          Neo4j Desktop                          │
│  1,385 vectors           43 nodes · 60+ relationships           │
│  BAAI/bge-small-en-v1.5  Company → Auditor → Subsidiary        │
│  9 SEC 10-K filings      → RiskFactor → Filing                 │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Routing Logic

The Planner node uses the LLM in JSON mode to classify every query:

| Route | Triggered by | Path |
|---|---|---|
| `calculate` | "Z-score", "ratio", "debt to equity" | Calculator tool → Responder |
| `graph` | "auditor", "subsidiary", "who", "related" | Neo4j traversal → Responder |
| `retrieve` | "what does the filing say", "risk factors" | Qdrant vector search → Responder |
| `direct` | Greetings, out-of-scope | Responder directly |

---

## Data

### Companies & Filings

| Company | Ticker | CIK | Filings | Why |
|---|---|---|---|---|
| Apple Inc. | AAPL | 0000320193 | 2023, 2024, 2025 10-K | Clean baseline — stable auditor (E&Y), healthy financials |
| SVB Financial Group | SIVB | 0000719739 | 2020, 2021, 2022 10-K | Bank failure March 2023 — interest rate risk visible in filings |
| Enron Corp | ENE | 0001024401 | 1998, 1999, 2000 10-K | Accounting fraud — SPE structures, Arthur Andersen collapse |

All filings fetched directly from **SEC EDGAR** (free, no API key required).

### Knowledge Graph

Built by `ingestion/build_graph.py` using curated, verified facts from public record:

```
(Enron Corp)-[:AUDITED_BY]->(Arthur Andersen)
(Enron Corp)-[:HAS_SUBSIDIARY]->(LJM Cayman LP {type: SPE})
(Enron Corp)-[:HAS_SUBSIDIARY]->(Raptor I {type: SPE})
(SVB Financial Group)-[:AUDITED_BY]->(KPMG)
(SVB Financial Group)-[:HAS_RISK]->(Interest Rate Risk {severity: critical})
(Apple Inc.)-[:AUDITED_BY]->(Ernst & Young)
```

### Key Findings

| Company | Metric | Value | Interpretation |
|---|---|---|---|
| Enron (2000) | Altman Z-score | **2.45** | Grey Zone — looked survivable on paper before collapse |
| SVB (2022) | Altman Z-score | **-0.70** | Deep Distress Zone — a year before the bank run |
| SVB (2022) | Current ratio | **0.16** | Critically low — $0.16 available per $1 of deposits |
| Apple (2023) | Altman Z-score | **8.14** | Safe Zone — structurally sound baseline |

---

## Evaluation

Evaluated against a golden dataset of 10 queries covering all three routes, using LLM-as-judge (faithfulness) and embedding cosine similarity (answer relevancy) — the same approach as the Ragas framework.

| Metric | Score | Target |
|---|---|---|
| **Faithfulness** | 0.806 | ≥ 0.70 |
| **Answer Relevancy** | 0.846 | ≥ 0.70 |
| **Topic Coverage** | 0.788 | — |
| **Routing Accuracy** | **10 / 10** | 10 / 10 |

Run evaluation:
```bash
python evaluation/ragas_eval.py
# Results saved to evaluation/results.json
```

---

## Project Structure

```
financial-anomaly-agent/
│
├── ingestion/
│   ├── fetch_edgar.py          # Pull 10-K filings from SEC EDGAR (no API key needed)
│   ├── chunk_and_embed.py      # Chunk → embed → load into Qdrant (1,385 vectors)
│   ├── build_graph.py          # Build Neo4j knowledge graph (curated, verified facts)
│   └── extract_entities.py     # LLM-based entity extraction from filing text (optional)
│
├── agent/
│   ├── state.py                # LangGraph AgentState schema
│   ├── llm.py                  # Groq API wrapper
│   ├── planner.py              # Route classification node (LLM → JSON)
│   ├── retriever.py            # Qdrant vector search + Neo4j Cypher traversal
│   ├── calculator_node.py      # Dispatches to financial ratio tools
│   ├── responder.py            # Memo synthesis with citations
│   └── graph.py                # LangGraph state machine + run_agent()
│
├── tools/
│   └── calculator.py           # Altman Z-score, current ratio, debt/equity (simpleeval)
│
├── api/
│   └── main.py                 # FastAPI: /investigate, /investigate/stream, /companies
│
├── evaluation/
│   ├── ragas_eval.py           # Faithfulness + answer relevancy evaluation
│   └── results.json            # Latest evaluation results
│
├── ui/
│   ├── src/
│   │   ├── App.jsx             # Root component
│   │   ├── App.css             # Dark theme styles
│   │   ├── api.js              # SSE streaming client
│   │   └── components/
│   │       ├── Sidebar.jsx     # Company cards + quick queries
│   │       ├── Chat.jsx        # Message thread
│   │       ├── Message.jsx     # User/agent/stage message types
│   │       ├── Input.jsx       # Auto-resizing textarea
│   │       ├── GraphViz.jsx    # D3 force-directed knowledge graph
│   │       └── Citations.jsx   # Source citations / graph panel
│   ├── index.html
│   ├── package.json
│   └── vite.config.js          # Proxy: /investigate → localhost:8000
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- [Neo4j Desktop](https://neo4j.com/download/) (free) — create a local DB, start it
- [Groq API key](https://console.groq.com) (free tier, no credit card)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/financial-anomaly-agent.git
cd financial-anomaly-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

QDRANT_PATH=./data/qdrant_store
RAW_FILINGS_DIR=./data/raw_filings
```

### 3. Run the ingestion pipeline

```bash
# Step 1: Fetch real 10-K filings from SEC EDGAR (free, no key needed)
python ingestion/fetch_edgar.py
# Downloads 9 filings → data/raw_filings/

# Step 2: Chunk, embed, and load into Qdrant
python ingestion/chunk_and_embed.py
# Creates 1,385 vectors → data/qdrant_store/

# Step 3: Build the Neo4j knowledge graph
python ingestion/build_graph.py --reset
# Creates 43 nodes + 60+ relationships in Neo4j
```

### 4. Start the backend

```bash
uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Start the frontend

```bash
cd ui
npm install
npm run dev
# UI: http://localhost:3000
```

### 6. (Optional) Run evaluation

```bash
python evaluation/ragas_eval.py
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/companies` | GET | List available companies |
| `/investigate` | POST | Run query, blocking response |
| `/investigate/stream` | POST | Run query, SSE streaming response |

### Example request

```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Altman Z-score for Enron in 2000?"}'
```

```json
{
  "query": "What is the Altman Z-score for Enron in 2000?",
  "route": "calculate",
  "answer": "Enron's Altman Z-score for 2000 is 2.45, placing it in the Grey Zone...",
  "citations": [],
  "latency_ms": 1842
}
```

---

## Design Decisions

**Why curated graph over LLM extraction?**
The knowledge graph is built from verified, authoritative facts (auditor assignments, known subsidiaries, documented risk factors) rather than free-text extraction. In production, institutional relationship data comes from structured sources like Bloomberg, Refinitiv, or EDGAR XBRL — not from parsing narrative text. The `extract_entities.py` script demonstrates the LLM extraction pipeline but is not used for the primary graph, which is the architecturally correct approach for well-known entities.

**Why Groq over OpenAI?**
Free tier with Llama 3.3 70B (500k TPD on 8b-instant, 100k on 70b). Zero cost for portfolio development, production-quality output for financial reasoning tasks.

**Why local Qdrant over cloud?**
`qdrant-client` in local file mode persists vectors on disk without a server process. Architecturally identical to Qdrant Cloud — swap `QdrantClient(path=...)` for `QdrantClient(url=..., api_key=...)` for production deployment.

**Why not Streamlit?**
The UI is a production React + Vite app with SSE streaming, D3 force graph, and component architecture. It demonstrates frontend engineering judgement, not just a data science notebook.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent framework | LangGraph 0.1.19 | Stateful multi-node agent with conditional routing |
| LLM | Groq (Llama 3.3 70B) | Planning, extraction, memo synthesis |
| Embeddings | BAAI/bge-small-en-v1.5 | Local embeddings, 384 dimensions, no API key |
| Vector store | Qdrant 1.9.1 | Semantic search over filing chunks |
| Graph DB | Neo4j 5 (Desktop) | Entity relationships with provenance |
| API | FastAPI 0.111 + SSE | Streaming REST API |
| Frontend | React 18 + Vite 5 | Chat UI with D3 graph visualization |
| Safe eval | simpleeval | Sandboxed financial ratio computation |
| Data source | SEC EDGAR | Free public 10-K filings, no API key |

---

## Roadmap

- [ ] Document upload — ingest any 10-K PDF and add to the graph automatically
- [ ] XBRL extraction — parse structured financial data for precise ratio calculation
- [ ] Cross-company comparison — "Compare SVB's risk profile with Enron's in their final year"
- [ ] Temporal graph — track how entity relationships change across filing years
- [ ] Locust load testing + Prometheus metrics (production observability layer)

---

## Author

**Avin Joseph**

---

## License

MIT

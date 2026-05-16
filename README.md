# Scientific Discovery Copilot

## Problem Statement

Modern science is drowning in papers but starving for synthesis.

Every week, thousands of research papers introduce new methods, biomarkers, datasets, contradictions, and unexplored connections. A human researcher can read a handful deeply, but the real breakthroughs often hide between papers: two studies that never cite each other, a method from one domain that could unlock another, or a weak signal repeated across disconnected literature.

**Scientific Discovery Copilot turns research papers into an active discovery engine.** It does not behave like a chatbot that answers from PDFs. It parses papers, extracts scientific concepts, builds a concept graph, detects cross-paper bridges, surfaces research gaps, and generates testable hypotheses grounded in evidence.

The goal is simple and ambitious: help researchers move from "What did these papers say?" to **"What should we investigate next?"**

## What We Built

This repository contains the finalized fast MVP of the Scientific Discovery Copilot, optimized to run locally on a CPU while still using Gemma through Ollama for reasoning tasks.

We originally explored a heavier architecture with PostgreSQL, Neo4j, Celery, Redis, and larger model paths. During testing, the fully heavy stack was too slow for local CPU-only demos, so we rebuilt the product around a faster working pipeline without losing the core discovery value.

The current version is clean, practical, and demo-ready:

- A FastAPI backend for paper ingestion, search, graph analytics, and hypothesis generation.
- A Streamlit frontend for manual testing, paper upload, graph visualization, and discovery workflows.
- A CPU-first Ollama/Gemma setup.
- A local graph engine powered by NetworkX.
- A fast deterministic semantic analyzer for upload-time concept and relationship extraction.
- Optional embedding-based retrieval for stronger semantic search when the machine can support it.

## Core Product Capabilities

### 1. Paper Upload And Processing

The backend accepts PDF research papers, parses them with PyMuPDF, extracts text by section, chunks the paper, and stores searchable evidence.

Current processing extracts:

- Title and paper metadata where available.
- Abstract/body/section text.
- Chunked evidence passages.
- Scientific concepts.
- Concept relationships.
- Themes, gaps, and discovery signals.
- Paper-level graph nodes.

### 2. Fast Scientific Concept Extraction

To keep paper uploads fast on CPU, the MVP uses a deterministic scientific semantic analyzer during ingestion.

It identifies useful scientific concepts such as:

- Diseases and conditions.
- Methods and algorithms.
- Biomarkers and measurements.
- Medical/scientific concepts.
- Dataset and evaluation terms.
- Cross-domain technical signals.

It also creates concept-to-concept relationship edges so the system can reason over more than plain text.

### 3. Knowledge Graph

The project builds a NetworkX `MultiDiGraph` where papers and concepts become connected nodes.

Graph layers include:

- `PAPER` nodes for uploaded research papers.
- Concept nodes for extracted scientific entities.
- `MENTIONS` edges from papers to concepts.
- Relationship edges between concepts.
- Bridge flags for concepts appearing across multiple papers.
- Cluster nodes for semantic concept communities.

This graph is exported as JSON for the frontend and can be visualized directly in the Streamlit app.

### 4. Cross-Paper Bridge Discovery

The system detects concepts that appear across multiple papers and uses them as bridges between otherwise separate studies.

This is one of the main discovery features:

```text
Paper A -> Shared Concept -> Paper B
```

Instead of only showing "similar documents", the app shows **why** papers are connected through scientific concepts.

### 5. Research Gap Detection

The graph engine searches for concept pairs that co-occur but do not yet have strong direct relationship edges.

These become candidate research gaps:

- Concepts that appear together but are not directly studied.
- Weakly connected ideas across papers.
- Possible unexplored mechanisms.
- Places where a hypothesis could be formed.

### 6. Hub Concepts And Clusters

The graph engine computes central concepts using graph analytics.

The dashboard surfaces:

- Hub concepts with high graph importance.
- Concept clusters.
- Bridge concepts across papers.
- Paper neighborhoods.
- Research gap candidates.

This helps a researcher quickly understand the structure of a small research corpus.

### 7. Search And Evidence Retrieval

The backend supports search over processed paper chunks.

The default path is CPU-friendly lexical retrieval so the app remains responsive without GPU support. For stronger semantic retrieval, the project can optionally enable MiniLM embeddings and ChromaDB with:

```powershell
$env:FAST_USE_MINILM="1"
```

### 8. Gemma-Powered Hypothesis Generation

Gemma through Ollama is used for higher-level reasoning tasks such as hypothesis generation and model readiness checks.

The hypothesis pipeline combines:

- User research query.
- Retrieved paper chunks.
- Graph bridges.
- Research gaps.
- Hub concepts.
- Paper-level context.

The output is designed to be more than a summary. It proposes testable research directions grounded in the uploaded papers.

### 9. Dynamic Streamlit Dashboard

The frontend provides a practical researcher-facing interface:

- Upload research papers.
- Check backend and Gemma status.
- Run search queries.
- Generate hypotheses.
- Explore discovery analytics.
- View graph exports.
- Visualize paper-to-concept and inter-paper concept connections.
- Inspect processed papers.

The dashboard is intentionally simple, fast, and demo-friendly.

## Architecture

```text
PDF Papers
   |
   v
FastAPI Backend
   |
   |-- PyMuPDF parser
   |-- section-aware chunker
   |-- deterministic semantic analyzer
   |-- lexical / optional embedding retrieval
   |-- NetworkX graph engine
   |-- Gemma via Ollama
   |
   v
Discovery Outputs
   |
   |-- searchable evidence
   |-- concept graph
   |-- bridges
   |-- gaps
   |-- hubs
   |-- clusters
   |-- hypotheses
   |
   v
Streamlit Frontend
```

## Project Layout

```text
scientific-discovery-copilot/
|-- backend/
|   |-- core/
|   |   |-- fast_gemma.py
|   |   |-- fast_graph.py
|   |   |-- pipeline.py
|   |   `-- semantic_analyzer.py
|   |-- ingestion/
|   |   |-- fast_chunker.py
|   |   `-- fast_parser.py
|   |-- retrieval/
|   |   |-- fast_embedder.py
|   |   `-- fast_vector_store.py
|   |-- main.py
|   |-- requirements.txt
|   |-- run_backend.ps1
|   `-- Dockerfile
|-- frontend/
|   |-- app.py
|   |-- requirements.txt
|   `-- README.md
|-- docker-compose.yml
|-- .gitignore
`-- README.md
```

## How To Run

### 1. Start Ollama On CPU

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot
docker compose up -d ollama
```

Pull Gemma if it is not already available:

```powershell
docker exec -it scientific-discovery-copilot-ollama-1 ollama pull gemma4:e2b
```

### 2. Run The Backend

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\run_backend.ps1
```

Backend URLs:

```text
http://127.0.0.1:8011
http://127.0.0.1:8011/docs
```

### 3. Run The Frontend

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\frontend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

Frontend URL:

```text
http://localhost:8501
```

## Docker Backend

The compose setup is CPU-first and includes Ollama plus the backend service.

```powershell
docker compose up -d ollama
docker compose up --build backend
```

Inside Docker, the backend talks to Ollama at:

```text
http://ollama:11434
```

## API Highlights

- `GET /health` - backend health check.
- `GET /model-status` - Ollama and Gemma reachability.
- `POST /warmup` - loads Gemma and confirms it responds.
- `POST /upload` - uploads and processes a PDF.
- `POST /search` - searches processed paper evidence.
- `POST /hypotheses` - generates evidence-grounded research hypotheses.
- `GET /discovery` - returns bridges, gaps, hubs, clusters, and graph stats.
- `GET /graph/export` - exports graph JSON for visualization.
- `GET /graph/bridges` - returns cross-paper bridge concepts.
- `GET /graph/gaps` - returns candidate research gaps.
- `GET /graph/hubs` - returns central concepts.
- `GET /papers` - lists processed papers.

## Model And Method Map

| Component | Method / Model | Role |
| --- | --- | --- |
| PDF parsing | PyMuPDF | Extracts text from uploaded papers |
| Chunking | Custom section-aware chunker | Splits papers into searchable evidence |
| Upload-time extraction | Deterministic semantic analyzer | Fast concept, relationship, theme, and gap extraction |
| Default retrieval | Lexical scoring | CPU-safe search with no model download |
| Optional retrieval | MiniLM + ChromaDB | Stronger semantic search when enabled |
| Graph analytics | NetworkX `MultiDiGraph` | Bridges, hubs, clusters, gaps, paper neighborhoods |
| Reasoning | Gemma via Ollama | Warmup, useful response checks, hypothesis generation |
| Interface | Streamlit | Researcher-facing dashboard and graph visualization |

## MVP Definition

The MVP proves the complete discovery loop:

1. Upload multiple papers.
2. Parse and chunk each paper.
3. Extract scientific concepts.
4. Build a graph connecting papers and concepts.
5. Search across the uploaded corpus.
6. Find cross-paper concept bridges.
7. Surface candidate research gaps.
8. Generate hypotheses grounded in evidence.
9. Visualize the graph and inter-paper concept connections.

## Planned WOW Factors

- Remote GPU Ollama support for larger Gemma models.
- Streaming generation with visible progress and time estimates.
- Stronger contradiction detection with paired paper claims.
- Citation-aware novelty scoring.
- Better biomedical NER when compute allows.
- Persistent multi-project storage.
- Exportable graph snapshots for reports and demos.
- A polished web frontend beyond Streamlit for production use.

## Why This Matters

Most AI paper tools help users read faster. Scientific Discovery Copilot aims higher: it helps users **think across papers**.

That difference matters. Reading tools summarize what is already written. Discovery tools reveal what is not yet connected.

This project is a working step toward AI systems that help researchers identify the next experiment, not just the next paragraph.

## Repository Hygiene

The repository has been cleaned to keep only the active product code:

- `backend/` is the source of truth for the FastAPI pipeline.
- `frontend/` is the source of truth for the Streamlit dashboard.
- Old prototype modules and duplicate fast folders have been removed from Git.
- Local-only folders such as `.venv`, `.hf-cache`, `data`, PDFs, and uploaded files are ignored.


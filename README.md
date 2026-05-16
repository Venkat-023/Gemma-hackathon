# Scientific Discovery Copilot

Fast CPU-friendly scientific discovery system for ingesting research papers, extracting concepts, building a knowledge graph, finding cross-paper bridges, and generating research hypotheses with Gemma through Ollama.

This repository now contains the finalized streamlined build. The older heavy Postgres/Neo4j/Celery prototype has been removed from Git. The active product is the working fast backend plus the Streamlit dashboard.

## Current Features

- PDF upload and parsing with PyMuPDF.
- Section-aware paper processing for abstract, methods, results, discussion, conclusion, and general body text.
- Fast chunking with overlap for semantic retrieval.
- CPU-first lexical retrieval enabled by default.
- Optional MiniLM embeddings and ChromaDB retrieval with `FAST_USE_MINILM=1`.
- Deterministic scientific concept extraction during upload for speed.
- Optional Gemma/Ollama calls for warmup, chat-style checks, and hypothesis generation.
- NetworkX `MultiDiGraph` knowledge graph.
- Paper nodes connected to concept nodes through `MENTIONS` edges.
- Concept-to-concept relationship edges from the semantic analyzer.
- Cross-paper bridge detection for concepts appearing across papers.
- Concept clusters, hub concepts, and research gap scoring.
- Inter-paper concept visualization: `Paper -> Shared Concept -> Paper`.
- Dynamic Streamlit dashboard for upload, search, hypotheses, discovery analytics, graph views, and paper inventory.

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

## Run Locally On CPU

Start Ollama:

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot
docker compose up -d ollama
```

Pull Gemma if needed:

```powershell
docker exec -it scientific-discovery-copilot-ollama-1 ollama pull gemma4:e2b
```

Create and install the backend environment:

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Run the backend:

```powershell
.\run_backend.ps1
```

Backend API:

```text
http://127.0.0.1:8011
http://127.0.0.1:8011/docs
```

Run the frontend:

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\frontend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

Frontend:

```text
http://localhost:8501
```

## Docker Backend

The compose file is CPU-first. It starts Ollama and can also run the backend container.

```powershell
docker compose up -d ollama
docker compose up --build backend
```

The backend container talks to Ollama through `http://ollama:11434`.

## Environment Flags

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` locally, `http://ollama:11434` in Docker | Ollama server URL |
| `GEMMA_FAST_MODEL` | `gemma4:e2b` | Gemma model used by the fast pipeline |
| `FAST_BACKEND_PORT` | `8011` | Backend API port |
| `FAST_USE_MINILM` | `0` | Set to `1` to use MiniLM embeddings and ChromaDB |
| `FAST_SKIP_GEMMA_ON_UPLOAD` | `1` | Keeps uploads fast by using deterministic extraction |
| `HF_HOME` | `backend/.hf-cache` | Hugging Face cache path |

## API Highlights

- `GET /health` - backend health.
- `GET /model-status` - Ollama/Gemma reachability.
- `POST /warmup` - loads Gemma and confirms it responds.
- `POST /upload` - upload and process a PDF.
- `POST /search` - semantic paper search.
- `POST /hypotheses` - generate hypotheses from retrieved evidence and graph context.
- `GET /discovery` - combined bridges, gaps, hubs, clusters, and graph stats.
- `GET /graph/export` - graph JSON for visualization.
- `GET /graph/bridges` - cross-paper bridge concepts.
- `GET /graph/gaps` - candidate research gaps.
- `GET /graph/hubs` - central concepts.
- `GET /papers` - processed paper inventory.

## What Each Model Or Method Does

- Gemma through Ollama: model warmup, hypothesis generation, and optional LLM-backed scientific reasoning.
- Deterministic semantic analyzer: fast upload-time entity, relationship, theme, and gap extraction.
- Lexical retrieval: default CPU-safe search path with no model download.
- MiniLM plus ChromaDB: optional embedding retrieval path for stronger semantic search.
- NetworkX graph engine: concept graph, bridge analytics, hub scoring, clusters, paper neighborhoods, and graph export.
- Streamlit: manual product testing, graph visualization, and result inspection.

## MVP

The MVP demonstrates a complete local scientific discovery loop:

1. Upload two or more research papers.
2. Extract concepts and relationships.
3. Build a knowledge graph.
4. Search across paper evidence.
5. Find shared concepts between papers.
6. Surface bridge concepts, gaps, hubs, and clusters.
7. Generate hypotheses grounded in the processed corpus.
8. Visualize paper-to-concept and inter-paper concept graphs.

## Planned WOW Factors

- Remote GPU Ollama support for larger Gemma models.
- Streaming hypothesis generation with progress and time estimates.
- Richer contradiction detection using paired evidence claims.
- Citation-aware novelty scoring.
- Exportable graph snapshots for demos and reports.
- Persistent project/session store beyond the current local JSON and Chroma state.
- Better biomedical NER integration when GPU or higher-memory CPU environments are available.

## Notes For Collaborators

- Do not commit `.venv`, `.hf-cache`, `data`, uploaded PDFs, or local model files.
- The old `fast_backend/` and `fast_frontend/` folders are ignored if they remain locally from development.
- Use `backend/` and `frontend/` as the source of truth.
